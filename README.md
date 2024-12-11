# A3C and PPO for Atari Breakout and Pong

This repository contains implementations of two reinforcement learning algorithms, **Asynchronous Advantage Actor-Critic (A3C)** and **Proximal Policy Optimization (PPO)**, to master Atari games **Breakout-v4** and **Pong-v4**. The project highlights the strengths and weaknesses of both algorithms in terms of training stability, performance, and computational efficiency.

## Authors
- **Avnish Patel**  
  Department of Electrical and Computer Engineering  
  [patel.avni@northeastern.edu](mailto:patel.avni@northeastern.edu)

- **Maulik Patel**  
  Department of Electrical and Computer Engineering  
  [patel.maul@northeastern.edu](mailto:patel.maul@northeastern.edu)

## Project Overview

Reinforcement learning (RL) is a technique used to train intelligent agents to solve dynamic tasks. This project demonstrates the effectiveness of PPO and A3C in learning optimal strategies for the Atari environments **Breakout** and **Pong**.

### Key Features
- **PPO**: Exhibits stable learning with a clipped surrogate objective for efficient policy updates.  
- **A3C**: Leverages asynchronous gradient descent with multi-core CPUs for faster training but requires extensive tuning for stability.

### Key Results
- **PPO** showed superior performance in stability and robustness across both environments.
- **A3C**, while computationally efficient, faced instability during extended training, highlighting its limitations.

## Usage

- Run the PPO.ipynb notebook to train PPO on Atari Breakout and Pong.
- For A3C, go to the A3C/ folder and run Main_Script.py file.
- If the user wants to modify the hyperparameters or training length, these are accessible through arguments.

## Repository Structure
```plaintext
.
├── A3C/                     # A3C implementation and experiments
├── PPO.ipynb                # PPO implementation and results
├── README.md                # Project description and usage instructions

