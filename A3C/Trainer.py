
import gymnasium as gym
import ale_py
import torch
import torch.nn.functional as F
import torch.optim as optim

from Environments import DownScale_42, Normalize


from A3C_Model import ActorCritic

        
        
def train(worker_id, config, net, step_tracker, access_lock, opt=None, 
            ep_rewards=None, ep_steps=None, pol_losses=None, val_losses=None, 
            pol_entropies=None, ep_counter=None):
    """
    Training function for asynchronous advantage actor-critic (A3C).
    """
    # Seed setup for reproducibility
    torch.manual_seed(config.seed + worker_id)

    # Initialize the environment
    game_env = gym.make(config.env_name)
    game_env = DownScale_42(game_env)
    game_env = Normalize(game_env)
    game_env.seed(config.seed + worker_id)

    # Setup model and optimizer
    worker_net = ActorCritic(game_env.observation_space.shape[0], game_env.action_space)
    if opt is None:
        opt = optim.Adam(net.parameters(), lr=config.lr)

    worker_net.train()

    # Reset environment and tracking variables
    obs, info = game_env.reset()
    obs = torch.from_numpy(obs)
    is_done = True

    ep_len = 0
    cum_reward = 0
    entropy_sum = 0
    policy_loss_val = 0
    value_loss_val = 0
    
    while True:
        # Termination check
        with access_lock:
            if ep_counter.value >= config.max_episodes or step_tracker.value >= config.max_steps:
                if ep_counter.value >= config.max_episodes: print("Episode Limit Reached!")
                if step_tracker.value >= config.max_steps: print("Step Limit Reached!")
                break
        
        # Sync the worker model with the global model
        worker_net.load_state_dict(net.state_dict())
        
        if is_done:
            mem_c = torch.zeros(1, 256)
            mem_h = torch.zeros(1, 256)
            if ep_len > 0:
                if ep_rewards is not None and ep_steps is not None:
                    with access_lock:
                        ep_rewards.append(cum_reward)
                        ep_steps.append(step_tracker.value)
                        
                        if pol_losses is not None and val_losses is not None:
                            pol_losses.append(policy_loss.item())
                            val_losses.append(value_loss.item())

                        if pol_entropies is not None:
                            pol_entropies.append(float(entropy_sum / ep_len))
                        
                cum_reward = 0
                ep_len = 0
                entropy_sum = 0
                policy_loss_val = 0
                value_loss_val = 0
                
                with access_lock:
                    ep_counter.value += 1
                    if ep_counter.value % 10 == 0:
                        print(f"Completed Episodes: {ep_counter.value}", flush=True)
        else:
            mem_c = mem_c.detach()
            mem_h = mem_h.detach()

        val_preds, logs_probs, reward_logs, entropy_logs = [], [], [], []

        for t in range(config.num_steps):
            ep_len += 1

            # Forward pass through the model
            pred_val, logit_out, (mem_h, mem_c) = worker_net((obs.unsqueeze(0), (mem_h, mem_c)))

            # Action and probability computations
            action_prob = F.softmax(logit_out, dim=-1)
            log_act_prob = F.log_softmax(logit_out, dim=-1)

            # Entropy calculation for exploration
            entropy_val = -(log_act_prob * action_prob).sum(1, keepdim=True)
            entropy_logs.append(entropy_val.item())
            entropy_sum += entropy_val.item()

            # Sample an action
            action = action_prob.multinomial(num_samples=1).detach()
            log_act_prob = log_act_prob.gather(1, action)

            # Take action in the environment
            obs, reward, done_signal, truncated, info = game_env.step(action.item())
            is_done = done_signal or truncated or ep_len >= config.max_episode_length

            reward = max(min(reward, 1), -1)
            cum_reward += reward

            with access_lock:
                step_tracker.value += 1

            if is_done:
                obs, info = game_env.reset()

            obs = torch.from_numpy(obs)
            val_preds.append(pred_val)
            logs_probs.append(log_act_prob)
            reward_logs.append(reward)

            if is_done:
                break

        # Bootstrap value
        future_val = torch.zeros(1, 1)
        if not is_done:
            pred_val, _, _ = worker_net((obs.unsqueeze(0), (mem_h, mem_c)))
            future_val = pred_val.detach()

        val_preds.append(future_val)
        policy_loss, value_loss, gae_val = 0, 0, torch.zeros(1, 1)

        # Compute loss using Generalized Advantage Estimation (GAE)
        for i in reversed(range(len(reward_logs))):
            future_val = config.gamma * future_val + reward_logs[i]
            adv_val = future_val - val_preds[i]
            value_loss += 0.5 * adv_val.pow(2)

            # Temporal difference error
            td_error = reward_logs[i] + config.gamma * val_preds[i + 1] - val_preds[i]
            gae_val = gae_val * config.gamma * config.gae_lambda + td_error

            policy_loss -= logs_probs[i] * gae_val.detach() + config.entropy_coef * entropy_logs[i]

        # Backpropagation and gradient updates
        opt.zero_grad()
        (policy_loss + config.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(worker_net.parameters(), config.max_grad_norm)

        # Sync gradients with the shared model
        for local_param, shared_param in zip(worker_net.parameters(), net.parameters()):
            if shared_param.grad is None:
                shared_param._grad = local_param.grad
        
        opt.step()


