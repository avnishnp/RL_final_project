
import gymnasium as gym
import ale_py
import torch
import torch.nn.functional as F
import torch.optim as optim

from Environments import DownScale_42, Normalize


from A3C_Model import ActorCritic
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None, 
          shared_rewards = None, shared_steps=None, shared_policy_losses=None, 
          shared_value_losses=None, shared_entropies = None, episode_counter = None):
    """
    Training function for asynchronous advantage actor-critic (A3C).
    
    Args:
        rank (int): Process rank for multiprocessing.
        args: Configuration and hyperparameters.
        shared_model: Global model shared across processes.
        counter: Global counter for total steps.
        lock: Multiprocessing lock for shared memory access.
        optimizer: Optimizer for model updates.
        shared_rewards, shared_steps, shared_policy_losses, shared_value_losses, shared_entropies: 
            Shared lists for tracking performance statistics.
        episode_counter: Global counter for completed episodes.
    """
    torch.manual_seed(args.seed + rank)

    # Preprocess Environemnt and intiialize
    env = gym.make(args.env_name)
    env = DownScale_42(env)
    env = Normalize(env)
    env.seed(args.seed + rank)


    # Initialize local model and optimizer
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    # Reset environment and initialize tracking variables
    state, info = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    total_reward = 0
    total_entropy = 0
    policy_loss_value = 0
    value_loss_value = 0
    
    while True:
        # Check Termination
        with lock:
            if episode_counter.value >= args.max_episodes or counter.value >= args.max_steps:
                if episode_counter.value >= args.max_episodes: print("Episode Limit Reached!")
                if counter.value >= args.max_steps: print("Step Limit Reached!")
                break
        
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
            if episode_length > 0:  # Avoid logging for the first iteration
                if shared_rewards is not None and shared_steps is not None:
                    with lock:
                        shared_rewards.append(total_reward)  # Store episode reward
                        shared_steps.append(counter.value)
                        if shared_policy_losses is not None and shared_value_losses is not None:
                            shared_policy_losses.append(policy_loss.item())  # Store policy loss
                            shared_value_losses.append(value_loss.item())
                        if shared_entropies is not None:
                            shared_entropies.append(float(total_entropy / episode_length))
                  
                  
                # Reset episode statistics
                total_reward = 0
                episode_length = 0
                total_entropy = 0
                policy_loss_value = 0
                value_loss_value = 0
                
                with lock:
                    episode_counter.value += 1
                    if episode_counter.value % 10 == 0:  # Print every 10 episodes
                        print(f"Total episodes completed: {episode_counter.value}", flush = True)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1

            # Forward pass through the model
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            
            # Compute action and log probabilities
            action_prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            
            # Compute policy entropy for exploration regularization
            entropy = -(log_prob * action_prob).sum(1, keepdim=True)
            entropies.append(entropy.item())
            total_entropy += entropy.item()

            # Sample Action and relvant log probability
            action = action_prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            # Take Action
            state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated or episode_length >= args.max_episode_length

            # state = torch.from_numpy(state)  
            # done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)
            
            total_reward += reward

            with lock:
                counter.value += 1

            if done:
                # episode_length = 0
                state, info = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        # Compute final value for bootstrapping
        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        
        # Generalized Advantage Estimation (GAE)
        gae = torch.zeros(1, 1)
        
        # Compute expected return and advantage
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            gae = gae * args.gamma * args.gae_lambda + (rewards[i] + args.gamma * values[i + 1] - values[i])

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        # Backpropagation and gradient clipping
        optimizer.zero_grad()
        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Synchronize gradients with the shared model
        ensure_shared_grads(model, shared_model)
        optimizer.step()