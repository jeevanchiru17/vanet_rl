"""
Deep Q-Learning Agent for VANET Traffic Congestion Control
Implements DQN with experience replay and target network
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import pickle

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_size, action_size, hidden_sizes=[256, 128]):
        super(DQNetwork, self).__init__()
        
        layers = []
        input_size = state_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for VANET routing decisions"""
    
    def __init__(self, state_size, action_size, config=None):
        """
        Initialize DQN Agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            config: Configuration dictionary
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Default configuration
        default_config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'buffer_size': 100000,
            'target_update_freq': 10,
            'hidden_sizes': [256, 128]
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Q-Networks
        self.policy_net = DQNetwork(
            state_size, 
            action_size, 
            self.config['hidden_sizes']
        ).to(self.device)
        
        self.target_net = DQNetwork(
            state_size, 
            action_size, 
            self.config['hidden_sizes']
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.config['learning_rate']
        )
        
        # Loss function (Huber loss for stability)
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.config['buffer_size'])
        
        # Exploration parameters
        self.epsilon = self.config['epsilon_start']
        self.epsilon_end = self.config['epsilon_end']
        self.epsilon_decay = self.config['epsilon_decay']
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (enables exploration)
        
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: best action from Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.config['batch_size']:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config['batch_size']
        )
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.config['gamma'] * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update training statistics
        self.training_step += 1
        self.losses.append(loss.item())
        
        # Update target network periodically
        if self.training_step % self.config['target_update_freq'] == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save_model(self, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'config': self.config,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.losses = checkpoint.get('losses', [])
        print(f"Model loaded from {filepath}")
    
    def get_statistics(self):
        """Get training statistics"""
        return {
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'buffer_size': len(self.replay_buffer)
        }


class VANETState:
    """State representation for VANET environment"""
    
    @staticmethod
    def extract_features(raw_state):
        """
        Extract features from raw NS3 state
        
        Expected raw_state format:
        {
            'traffic_density': float,  # vehicles per km
            'avg_queue_length': float,  # packets
            'link_quality': float,  # SINR in dB
            'vehicle_speed': float,  # m/s
            'congestion_level': float,  # 0-1
            'packet_loss_rate': float,  # 0-1
            'neighbor_count': int,
            'distance_to_destination': float  # meters
        }
        
        Returns:
            Normalized feature vector
        """
        features = [
            raw_state.get('traffic_density', 0) / 100.0,  # Normalize by max expected
            raw_state.get('avg_queue_length', 0) / 50.0,
            (raw_state.get('link_quality', 0) + 10) / 40.0,  # SINR: -10 to 30 dB
            raw_state.get('vehicle_speed', 0) / 30.0,  # Max ~30 m/s
            raw_state.get('congestion_level', 0),
            raw_state.get('packet_loss_rate', 0),
            raw_state.get('neighbor_count', 0) / 20.0,
            min(raw_state.get('distance_to_destination', 0) / 1000.0, 1.0)
        ]
        
        return np.array(features, dtype=np.float32)


class VANETReward:
    """Reward function for VANET routing"""
    
    @staticmethod
    def calculate_reward(metrics):
        """
        Calculate reward based on network performance metrics
        
        Args:
            metrics: Dictionary containing:
                - packet_delivered: bool
                - delay: float (seconds)
                - congestion_created: float (0-1)
                - routing_overhead: int (control packets)
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Reward for successful delivery
        if metrics.get('packet_delivered', False):
            reward += 10.0
        else:
            reward -= 5.0
        
        # Penalty for delay
        delay = metrics.get('delay', 0)
        if delay > 0:
            reward -= min(delay * 2, 5.0)  # Cap penalty at -5
        
        # Penalty for congestion
        congestion = metrics.get('congestion_created', 0)
        reward -= congestion * 3.0
        
        # Penalty for routing overhead
        overhead = metrics.get('routing_overhead', 0)
        reward -= overhead * 0.1
        
        return reward


if __name__ == "__main__":
    # Example usage
    print("DQN Agent for VANET Traffic Congestion Control")
    print("=" * 50)
    
    # Initialize agent
    state_size = 8  # Number of state features
    action_size = 5  # Number of routing actions (e.g., 5 possible next hops)
    
    agent = DQNAgent(state_size, action_size)
    
    print(f"\nAgent initialized:")
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Device: {agent.device}")
    print(f"\nPolicy Network:")
    print(agent.policy_net)
    
    # Test state extraction
    raw_state = {
        'traffic_density': 50,
        'avg_queue_length': 10,
        'link_quality': 15,
        'vehicle_speed': 20,
        'congestion_level': 0.3,
        'packet_loss_rate': 0.05,
        'neighbor_count': 8,
        'distance_to_destination': 500
    }
    
    state = VANETState.extract_features(raw_state)
    print(f"\nExample state vector: {state}")
    
    # Test action selection
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    
    # Test reward calculation
    metrics = {
        'packet_delivered': True,
        'delay': 0.05,
        'congestion_created': 0.2,
        'routing_overhead': 3
    }
    reward = VANETReward.calculate_reward(metrics)
    print(f"Calculated reward: {reward}")
