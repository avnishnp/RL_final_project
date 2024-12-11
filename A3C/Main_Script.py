# from __future__ import print_function

import argparse
import os
from Environments import DownScale_42, Normalize
from A3C_Model import ActorCritic
from Trainer import train
import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
import copy

import matplotlib.pyplot as plt
from Optimizer import Adam


import gymnasium as gym
import ale_py

from datetime import datetime


def save_arguments(args, dir_path):
    """Save the arguments to a text file."""
    args_dict = vars(args)
    args_file_path = os.path.join(dir_path, "arguments.txt")
    with open(args_file_path, "w") as f:
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")
    print(f"Arguments saved in: {args_file_path}")
    
def create_directory_structure(base_dir, sweep_name, hyperparam_name, value):
    """Create directory structure for hyperparameter sweeps."""
    mega_sweep_dir = os.path.join(base_dir, sweep_name)
    os.makedirs(mega_sweep_dir, exist_ok=True)
    
    if hyperparam_name is None:
        # Base directory for baseline
        run_dir = os.path.join(mega_sweep_dir, "base")
    else:
        # Directory for specific hyperparameter value
        run_dir = os.path.join(mega_sweep_dir, f"{hyperparam_name}_{value}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, mega_sweep_dir

def moving_average(data, window_size):
        return [
            sum(data[i:i + window_size]) / min(window_size, len(data[i:i + window_size]))
            for i in range(len(data))
    ]



if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    parser = argparse.ArgumentParser(description='A3C')

    #Set Training Lengths
    parser.add_argument('--env-name', default='Breakout-v4')
    parser.add_argument('--no-shared', default=False)
    parser.add_argument('--run-base-case', type=lambda x: x.lower() == 'true', default=True)    
    parser.add_argument('--seed', type=int, default=1)


    parser.add_argument('--max-episode-length', type=int, default=100000)
    parser.add_argument('--max-episodes', type=int, default=500)
    parser.add_argument('--max-steps', type=int, default=20000000)

    parser.add_argument('--num-processes', type=int, default=8)


    # Tuning Arguments
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--entropy-coef', type=float, default=0.15)
    parser.add_argument('--num-steps', type=int, default=80)
    parser.add_argument('--max-grad-norm', type=float, default=5)

    # Extra Arguments
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=1.00)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)



    args = parser.parse_args()

    # Base Directory
    print(args)
    base_dir = "MegaSweep"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_name = f"Sweep_{current_time}"

    # Define hyperparameter sweep
    hyperparams_to_sweep = {
        # "lr": [
            # 0.00001, 
            # 0.0005],
            # "entropy-coef": [.05]
            # "num-steps":[10]
    }
    print(hyperparams_to_sweep)
    
    episode_counter = mp.Value('i', 0)  
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    
    manager = Manager()
    shared_rewards = manager.list()
    shared_steps = manager.list()
    shared_policy_losses = manager.list()   
    shared_value_losses = manager.list()
    shared_entropies = manager.list() 
    
    
    torch.manual_seed(args.seed)
    
    env = gym.make(args.env_name)
    env = DownScale_42(env)
    env = Normalize(env)
    
    
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = Adam(shared_model.parameters(), lr=args.lr)
        optimizer.global_memory()


    # Run Base Configuration
    if args.run_base_case:
        base_run_dir, mega_sweep_dir = create_directory_structure(base_dir, sweep_name, None, None)
        save_arguments(args, base_run_dir)


        processes = []
       
        for rank in range(0, args.num_processes-1):
            p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, 
                                            optimizer, shared_rewards, shared_steps, 
                                            shared_policy_losses, shared_value_losses, 
                                            shared_entropies, episode_counter))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        print("All processes have stopped. Training complete.")

        print(f"Total episodes completed: {episode_counter.value}")
        print(f"Total steps completed: {counter.value}")
        
        rewards_list = list(shared_rewards)
        steps_list = list(shared_steps)
        policy_loss_list = list(shared_policy_losses)
        value_loss_list = list(shared_value_losses)
        entropy_list = list(shared_entropies)
        loss_totals = [(p + v) for p, v in zip(policy_loss_list, value_loss_list)]

        
        # Plot rewards vs episodes
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(entropy_list)), entropy_list, label="Entropy")
        plt.xlabel("Episodes")
        plt.ylabel("Entropy")
        plt.title("Entropy vs Episodes")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "entropy_vs_episodes.png"))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(rewards_list, label="Episode vs Rewards")
        plt.xlabel("Episodes Completed")
        plt.ylabel("Reward")
        plt.title("Learning Curve: Rewards vs Episodes")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "rewards_vs_episodes.png"))
        plt.close()

        # Plot rewards vs steps
        plt.figure(figsize=(12, 6))
        plt.plot(steps_list, rewards_list, label="Rewards vs Steps")
        plt.xlabel("Cumulative Steps")
        plt.ylabel("Reward")
        plt.title("Learning Curve: Rewards vs Steps")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "rewards_vs_steps.png"))
        plt.close()

        # Plot policy and value loss vs episodes
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(loss_totals)), loss_totals, label="Policy Loss vs Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Loss Curve: Total Loss vs Episodes")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "loss_vs_episodes.png"))
        plt.close()

        # Plot policy and value loss vs steps
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(policy_loss_list)), policy_loss_list, label="Policy Loss vs Episodes")
        plt.plot(range(len(value_loss_list)), value_loss_list, label="Value Loss vs Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Loss Curve: Policy and Value Loss vs Episodes")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "each_loss_vs_eps.png"))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps_list, policy_loss_list, label="Policy Loss vs Steps")
        plt.plot(steps_list, value_loss_list, label="Value Loss vs Steps")
        plt.xlabel("Cumulative Steps")
        plt.ylabel("Loss")
        plt.title("Loss Curve: Policy and Value Loss vs Steps")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "each_loss_vs_steps.png"))
        plt.close()
        
        
            # Apply moving average smoothing
        smoothed_rewards = moving_average(rewards_list, window_size=15)
        smoothed_loss_totals = moving_average(loss_totals, window_size=15)
        smoothed_value_loss = moving_average(value_loss_list, window_size=15)
        smoothed_policy_loss = moving_average(policy_loss_list, window_size = 15)
        smoothed_entropy = moving_average(entropy_list, window_size=15)


        # Plot rewards vs episodes
        plt.figure(figsize=(12, 6))
        plt.plot(smoothed_rewards, label="Rewards vs Episodes")
        plt.xlabel("Episodes Completed")
        plt.ylabel("Reward")
        plt.title("Learning Curve: Rewards vs Episodes")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "smoothed_rewards_vs_episodes.png"))
        plt.close()

        # Plot loss differences vs episodes
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(smoothed_loss_totals)), smoothed_loss_totals, label= "Loss vs Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Loss Curve: Total Loss vs Episodes")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "smoothed_loss_vs_episodes.png"))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(smoothed_value_loss)), smoothed_value_loss, label= "Value Loss vs Episodes")
        plt.plot(range(len(smoothed_policy_loss)), smoothed_policy_loss, label= "Policy Loss vs Episodes")    
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.title("Loss Curve: Each Loss vs Episodes")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "each smoothed_loss_vs_episodes.png"))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(smoothed_entropy)), smoothed_entropy, label="Entropy")
        plt.xlabel("Episodes")
        plt.ylabel("Entropy")
        plt.title("Entropy vs Episodes")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(base_run_dir, "smoothed entropy_vs_episodes.png"))
        plt.close()
        
    
    for param, values in hyperparams_to_sweep.items():
        for value in values:
            print(f"Running sweep for {param} = {value}")
            
            # Modify args for current hyperparameter
            sweep_args = copy.deepcopy(args)
            setattr(sweep_args, param, value)

            run_dir, _ = create_directory_structure(base_dir, sweep_name, param, value)
            save_arguments(sweep_args, run_dir)

            # Reset shared variables
            episode_counter.value = 0
            counter.value = 0
            shared_rewards[:] = []
            shared_steps[:] = []
            shared_policy_losses[:] = []
            shared_value_losses[:] = []
            shared_entropies[:] = []

            # Training for the current hyperparameter value
            torch.manual_seed(sweep_args.seed)
            processes = []

            for rank in range(0, sweep_args.num_processes - 1):
                p = mp.Process(target=train, args=(rank, sweep_args, shared_model, counter, lock, optimizer, 
                                                   shared_rewards, shared_steps, shared_policy_losses, 
                                                   shared_value_losses, shared_entropies, episode_counter))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                
            
                
            print("All processes have stopped. Training complete.")

            print(f"Total episodes completed: {episode_counter.value}")
            print(f"Total steps completed: {counter.value}")
            
            rewards_list = list(shared_rewards)
            steps_list = list(shared_steps)
            policy_loss_list = list(shared_policy_losses)
            value_loss_list = list(shared_value_losses)
            entropy_list = list(shared_entropies)
            loss_totals = [(p + v) for p, v in zip(policy_loss_list, value_loss_list)]

            
            # Plot rewards vs episodes
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(entropy_list)), entropy_list, label="Entropy")
            plt.xlabel("Episodes")
            plt.ylabel("Entropy")
            plt.title("Entropy vs Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "entropy_vs_episodes.png"))
            plt.close()
            
            plt.figure(figsize=(12, 6))
            plt.plot(rewards_list, label="Episode vs Rewards")
            plt.xlabel("Episodes Completed")
            plt.ylabel("Reward")
            plt.title("Learning Curve: Rewards vs Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "rewards_vs_episodes.png"))
            plt.close()

            # Plot rewards vs steps
            plt.figure(figsize=(12, 6))
            plt.plot(steps_list, rewards_list, label="Rewards vs Steps")
            plt.xlabel("Cumulative Steps")
            plt.ylabel("Reward")
            plt.title("Learning Curve: Rewards vs Steps")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "rewards_vs_steps.png"))
            plt.close()

            # Plot policy and value loss vs episodes
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(loss_totals)), loss_totals, label="Policy Loss vs Episodes")
            plt.xlabel("Episodes")
            plt.ylabel("Loss")
            plt.title("Loss Curve: Total Loss vs Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "loss_vs_episodes.png"))
            plt.close()

            # Plot policy and value loss vs steps
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(policy_loss_list)), policy_loss_list, label="Policy Loss vs Episodes")
            plt.plot(range(len(value_loss_list)), value_loss_list, label="Value Loss vs Episodes")
            plt.xlabel("Episodes")
            plt.ylabel("Loss")
            plt.title("Loss Curve: Policy and Value Loss vs Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "each_loss_vs_eps.png"))
            plt.close()
            
            plt.figure(figsize=(12, 6))
            plt.plot(steps_list, policy_loss_list, label="Policy Loss vs Steps")
            plt.plot(steps_list, value_loss_list, label="Value Loss vs Steps")
            plt.xlabel("Cumulative Steps")
            plt.ylabel("Loss")
            plt.title("Loss Curve: Policy and Value Loss vs Steps")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "each_loss_vs_steps.png"))
            plt.close()
            
            
                # Apply moving average smoothing
            smoothed_rewards = moving_average(rewards_list, window_size=15)
            smoothed_loss_totals = moving_average(loss_totals, window_size=15)
            smoothed_value_loss = moving_average(value_loss_list, window_size=15)
            smoothed_policy_loss = moving_average(policy_loss_list, window_size = 15)
            smoothed_entropy = moving_average(entropy_list, window_size=15)


            # Plot rewards vs episodes
            plt.figure(figsize=(12, 6))
            plt.plot(smoothed_rewards, label="Rewards vs Episodes")
            plt.xlabel("Episodes Completed")
            plt.ylabel("Reward")
            plt.title("Learning Curve: Rewards vs Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "smoothed_rewards_vs_episodes.png"))
            plt.close()

            # Plot loss differences vs episodes
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(smoothed_loss_totals)), smoothed_loss_totals, label= "Loss vs Episodes")
            plt.xlabel("Episodes")
            plt.ylabel("Loss")
            plt.title("Loss Curve: Total Loss vs Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "smoothed_loss_vs_episodes.png"))
            plt.close()
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(smoothed_value_loss)), smoothed_value_loss, label= "Value Loss vs Episodes")
            plt.plot(range(len(smoothed_policy_loss)), smoothed_policy_loss, label= "Policy Loss vs Episodes")    
            plt.xlabel("Episodes")
            plt.ylabel("Loss")
            plt.title("Loss Curve: Each Loss vs Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "each smoothed_loss_vs_episodes.png"))
            plt.close()
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(smoothed_entropy)), smoothed_entropy, label="Entropy")
            plt.xlabel("Episodes")
            plt.ylabel("Entropy")
            plt.title("Entropy vs Episodes")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(run_dir, "smoothed entropy_vs_episodes.png"))
            plt.close()
    
    
    


    

    print(f"Hyperparameter sweep complete. Results saved in {mega_sweep_dir}.")
