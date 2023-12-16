# MLSec-Project

## Group Members:

- Anjali Gohil
- Hariharan N
- Prashanth Rebala
- Rushang Gajjal

## Project Idea

- Penetration Testing plays a critical role in evaluating the security of target systems by emulating real active adversaries.

- However, the current approach demands significant manual effort, particularly for expansive and intricate networks, leading to outcomes heavily reliant on the expertise of pen-testers, thus diminishing repeatability.

## Methodology and Approach

- Network Attack Simulator creates a detailed simulation of a real-life network topology and infrastructure

- Scenario definition consists of: Network configuration, Host configurations and pen-tester configurations

- Supports partially observable mode; reflecting the reality of pen-testing more accurately.

- To address the challenge of achieving multiple objectives we model the solution as an Multi-Objective Markov Decision Process i.e. the reward R is a vector with n individual rewards, instead of a scalar reward.

- We employ Proximal Policy Optimization (PPO) as it exhibits stable responsiveness to environmental changes, adjusting the gradient update step size optimally and promoting exploration.

- These algorithms will train intelligent agents to maximize control over systems within a complex state space that simulates network topology.
