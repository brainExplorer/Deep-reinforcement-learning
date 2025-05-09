DQN for CartPole using PyTorch
This project implements a Deep Q-Network (DQN) using PyTorch to solve the CartPole-v1 environment from OpenAI Gym. It includes experience replay, target network updates, and epsilon-greedy action selection.
Features
- Deep Q-Network architecture with 2 hidden layers
- Experience replay buffer
- Target network for stability
- Epsilon-greedy policy for exploration vs exploitation
- CartPole-v1 environment from OpenAI Gymnasium
Installation
Install the required libraries:
pip install torch gymnasium matplotlib
Usage
Run the training script:
python dqn_cartpole.py
Project Structure
dqn_cartpole.py - Main script implementing the DQN agent and training loop
Training Details
The model is trained over 250 episodes using a replay memory of 10,000 transitions and a batch size of 128. The epsilon-greedy strategy is used for action selection with decaying epsilon. Target network is softly updated after each step using a factor tau = 0.005.
License
This project is open-source and available under the MIT License.
