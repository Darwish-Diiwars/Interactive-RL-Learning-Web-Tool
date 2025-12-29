# Interactive RL Learning Web Tool

**Author:** Mohamed Ghalwash  
**Course:** Reinforcement Learning (DSAI 402)  
**Institution:** Zewail City School of Computational and Artificial Intelligence  
**Date:** December 2025  
**Bonus Competition Submission** (as per Prof. Mohamed Ghalwash's rules, Dec 11, 2025)

## Project Overview
This is a web-based interactive tool for learning reinforcement learning (RL), built as a submission for the **Competition Bonus** in the Reinforcement Learning course.

The tool provides:
- An intuitive interface to select environments and algorithms.
- Real-time visualization of state, agent's actions (policy arrows), and value function (heatmap).
- Narrator logs explaining algorithm updates (e.g., delta, Q/V changes).

All algorithms are implemented **from scratch** based on pseudocode from:
- **Reinforcement Learning: An Introduction** by Sutton & Barto (2nd Edition, 2018)
- Lecture slides provided by Prof. Mohamed Ghalwash (2025–2026)

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python app.py`
3. Open http://localhost:5000 in your browser.

**Objective**
Build a web-based interactive tool for learning reinforcement learning (RL). The tool is mainly for anyone who is interested in RL. It provides an intuitive interface for exploring key algorithms, environments, and concepts.

**Requirements**
- **Environments**: Include a variety of RL environments such as GridWorld, CartPole, MountainCar, FrozenLake, and custom environments (e.g., Breakout and Gym4Real). Each environment should be visualized to show the agent's actions and the environment's state in real time.
- **Algorithms**: Implement and visualize the following RL algorithms:
  - Policy Iteration and Policy Evaluation
  - Value Iteration
  - Monte Carlo (MC)
  - Temporal Difference (TD)

## Algorithms Implemented (with Sources)

| Algorithm                  | Source (Sutton & Barto 2nd ed.) | Lecture Slide Reference |
|----------------------------|----------------------------------|--------------------------|
| Policy Iteration           | Figure 4.2                       | Policy Iteration         |
| Value Iteration            | Figure 4.3                       | Value Iteration          |
| Monte Carlo Prediction     | Figure 5.1 (first-visit)         | Monte Carlo Prediction   |
| e-greedy Monte Carlo       | Chapter 5                        | e-greedy MC              |
| Temporal Difference (TD(0))| Figure 6.1                       | Temporal Difference      |
| n-step TD                  | Chapter 7                        | n-step TD                |
| SARSA                      | Figure 6.4                       | SARSA                    |
| Q-Learning                 | Figure 6.5                       | Q-Learning               |

## Environments Used

| Environment              | Type       | Description |
|--------------------------|------------|-------------|
| StochasticGridWorld      | Custom     | 6×6 grid with stochastic slips (0.1) and obstacles |
| Taxi-v3                  | Gymnasium  | Discrete taxi environment (500 states) |
| CartPole-v1              | Gymnasium  | Continuous state (discretized to 10×10×10×10 bins) |

## Requirements.txt
