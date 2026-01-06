"""
Training script for VANET DQN Agent
Integrates NS3 simulation with Python DQN agent
"""

import os
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent, VANETState, VANETReward
import pandas as pd
import json
from datetime import datetime


class NS3Environment:
    """Interface to NS3 VANET simulation"""
    
    def __init__(self, ns3_path, output_dir="vanet-dqn-results"):
        self.ns3_path = ns3_path
        self.output_dir = output_dir
        self.state_file = os.path.join(output_dir, "state.csv")
        self.action_file = os.path.join(output_dir, "action.csv")
        self.reward_file = os.path.join(output_dir, "reward.csv")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Simulation parameters
        self.sim_params = {
            'nVehicles': 50,
            'simTime': 100.0,
            'areaSize': 1000.0,
            'packetSize': 1024,
            'packetInterval': 0.1
        }
    
    def set_params(self, **kwargs):
        """Update simulation parameters"""
        self.sim_params.update(kwargs)
    
    def write_action(self, action):
        """Write action for NS3 to read"""
        with open(self.action_file, 'w') as f:
            f.write(str(action))
    
    def read_state(self):
        """Read state from NS3"""
        if not os.path.exists(self.state_file):
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                line = f.readline().strip()
                if not line:
                    return None
                values = [float(x) for x in line.split(',')]
                
                raw_state = {
                    'traffic_density': values[0],
                    'avg_queue_length': values[1],
                    'link_quality': values[2],
                    'vehicle_speed': values[3],
                    'congestion_level': values[4],
                    'packet_loss_rate': values[5],
                    'neighbor_count': int(values[6]),
                    'distance_to_destination': values[7]
                }
                return VANETState.extract_features(raw_state)
        except Exception as e:
            print(f"Error reading state: {e}")
            return None
    
    def read_reward(self):
        """Read reward metrics from NS3"""
        if not os.path.exists(self.reward_file):
            return None
        
        try:
            with open(self.reward_file, 'r') as f:
                line = f.readline().strip()
                if not line:
                    return None
                values = line.split(',')
                
                metrics = {
                    'packet_delivered': bool(int(values[0])),
                    'delay': float(values[1]),
                    'congestion_created': float(values[2]),
                    'routing_overhead': int(values[3])
                }
                return VANETReward.calculate_reward(metrics)
        except Exception as e:
            print(f"Error reading reward: {e}")
            return None
    
    def run_episode(self, agent, training=True):
        """Run one simulation episode"""
        # Build command
        cmd = [
            os.path.join(self.ns3_path, "ns3"),
            "run",
            "vanet-dqn-simulation",
            "--"
        ]
        
        for key, value in self.sim_params.items():
            cmd.extend([f"--{key}={value}"])
        
        cmd.extend([f"--outputDir={self.output_dir}"])
        
        print(f"Running NS3 simulation: {' '.join(cmd)}")
        
        # Clear previous state/reward files
        for f in [self.state_file, self.reward_file]:
            if os.path.exists(f):
                os.remove(f)
        
        # Initialize action
        self.write_action(0)
        
        # Run simulation in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.ns3_path
        )
        
        episode_reward = 0
        step_count = 0
        
        # Monitor simulation and interact with agent
        while process.poll() is None:
            # Check for new state
            state = self.read_state()
            if state is not None:
                # Agent selects action
                action = agent.select_action(state, training=training)
                self.write_action(action)
                
                # Wait a bit for NS3 to process
                time.sleep(0.1)
                
                # Read reward
                reward = self.read_reward()
                if reward is not None:
                    episode_reward += reward
                    step_count += 1
                    
                    # Store experience (simplified - in real implementation, 
                    # would need next_state)
                    if training and step_count > 1:
                        # This is a simplified version
                        # In practice, you'd need to properly track state transitions
                        pass
            
            time.sleep(0.5)  # Poll every 500ms
        
        # Wait for process to complete
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"NS3 simulation failed with return code {process.returncode}")
            print(f"Error: {stderr.decode()}")
            return None, 0
        
        print(f"Episode completed: {step_count} steps, total reward: {episode_reward:.2f}")
        
        return episode_reward, step_count
    
    def get_flow_stats(self):
        """Read flow statistics from NS3 output"""
        stats_file = os.path.join(self.output_dir, "flow-stats.txt")
        if not os.path.exists(stats_file):
            return None
        
        stats = {}
        with open(stats_file, 'r') as f:
            content = f.read()
            
            # Parse statistics (simplified)
            for line in content.split('\n'):
                if 'Average Throughput:' in line:
                    stats['throughput'] = float(line.split(':')[1].strip().split()[0])
                elif 'Average Delay:' in line:
                    stats['delay'] = float(line.split(':')[1].strip().split()[0])
                elif 'Packet Delivery Ratio:' in line:
                    stats['pdr'] = float(line.split(':')[1].strip().split()[0])
        
        return stats


def train_dqn_agent(episodes=100, save_interval=10):
    """Main training loop"""
    
    print("=" * 70)
    print("VANET Traffic Congestion Control - DQN Training")
    print("=" * 70)
    
    # Initialize environment
    ns3_path = "/Users/jeevanhr/vanet_RL/ns-3.46"
    output_dir = "vanet-dqn-results"
    env = NS3Environment(ns3_path, output_dir)
    
    # Initialize agent
    state_size = 8
    action_size = 5  # 5 possible routing decisions
    
    config = {
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
    
    agent = DQNAgent(state_size, action_size, config)
    
    # Training statistics
    episode_rewards = []
    episode_steps = []
    throughputs = []
    delays = []
    pdrs = []
    
    # Create results directory
    results_dir = "training_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Training loop
    for episode in range(episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{episodes}")
        print(f"{'='*70}")
        
        # Vary simulation parameters for diverse training
        if episode % 10 == 0:
            n_vehicles = np.random.randint(30, 70)
            env.set_params(nVehicles=n_vehicles)
            print(f"Updated parameters: nVehicles={n_vehicles}")
        
        # Run episode
        episode_reward, steps = env.run_episode(agent, training=True)
        
        if episode_reward is None:
            print("Episode failed, skipping...")
            continue
        
        episode_rewards.append(episode_reward)
        episode_steps.append(steps)
        
        # Get flow statistics
        stats = env.get_flow_stats()
        if stats:
            throughputs.append(stats.get('throughput', 0))
            delays.append(stats.get('delay', 0))
            pdrs.append(stats.get('pdr', 0))
            
            print(f"\nPerformance Metrics:")
            print(f"  Throughput: {stats.get('throughput', 0):.2f} kbps")
            print(f"  Delay: {stats.get('delay', 0):.4f} s")
            print(f"  PDR: {stats.get('pdr', 0):.2f}%")
        
        # Train agent (in real implementation, this would use collected experiences)
        if len(agent.replay_buffer) >= agent.config['batch_size']:
            for _ in range(10):  # Multiple training steps per episode
                loss = agent.train_step()
                if loss is not None:
                    print(f"Training loss: {loss:.4f}")
        
        # Print agent statistics
        agent_stats = agent.get_statistics()
        print(f"\nAgent Statistics:")
        print(f"  Epsilon: {agent_stats['epsilon']:.4f}")
        print(f"  Buffer size: {agent_stats['buffer_size']}")
        print(f"  Training steps: {agent_stats['training_step']}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(results_dir, f"dqn_model_ep{episode+1}.pth")
            agent.save_model(model_path)
            print(f"\nModel saved to {model_path}")
        
        # Plot progress
        if (episode + 1) % 5 == 0:
            plot_training_progress(episode_rewards, throughputs, delays, pdrs, results_dir)
    
    # Save final model
    final_model_path = os.path.join(results_dir, "dqn_model_final.pth")
    agent.save_model(final_model_path)
    
    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'throughputs': throughputs,
        'delays': delays,
        'pdrs': pdrs,
        'config': config
    }
    
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training history saved to: {history_path}")
    print(f"{'='*70}")
    
    return agent, history


def plot_training_progress(rewards, throughputs, delays, pdrs, output_dir):
    """Plot training progress"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('VANET DQN Training Progress', fontsize=16)
    
    # Episode rewards
    axes[0, 0].plot(rewards, label='Episode Reward', color='blue', alpha=0.6)
    if len(rewards) > 10:
        window = min(10, len(rewards))
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        axes[0, 0].plot(moving_avg, label=f'{window}-Episode Moving Avg', 
                       color='red', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Throughput
    if throughputs:
        axes[0, 1].plot(throughputs, label='Throughput', color='green', alpha=0.6)
        if len(throughputs) > 10:
            window = min(10, len(throughputs))
            moving_avg = pd.Series(throughputs).rolling(window=window).mean()
            axes[0, 1].plot(moving_avg, label=f'{window}-Episode Moving Avg', 
                           color='darkgreen', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Throughput (kbps)')
        axes[0, 1].set_title('Network Throughput')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Delay
    if delays:
        axes[1, 0].plot(delays, label='Delay', color='orange', alpha=0.6)
        if len(delays) > 10:
            window = min(10, len(delays))
            moving_avg = pd.Series(delays).rolling(window=window).mean()
            axes[1, 0].plot(moving_avg, label=f'{window}-Episode Moving Avg', 
                           color='darkorange', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Delay (s)')
        axes[1, 0].set_title('End-to-End Delay')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Packet Delivery Ratio
    if pdrs:
        axes[1, 1].plot(pdrs, label='PDR', color='purple', alpha=0.6)
        if len(pdrs) > 10:
            window = min(10, len(pdrs))
            moving_avg = pd.Series(pdrs).rolling(window=window).mean()
            axes[1, 1].plot(moving_avg, label=f'{window}-Episode Moving Avg', 
                           color='indigo', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('PDR (%)')
        axes[1, 1].set_title('Packet Delivery Ratio')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'training_progress_{len(rewards)}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Progress plot saved to {plot_path}")


def evaluate_agent(model_path, num_episodes=10):
    """Evaluate trained agent"""
    
    print("=" * 70)
    print("VANET DQN Agent Evaluation")
    print("=" * 70)
    
    # Initialize environment
    ns3_path = "/Users/jeevanhr/vanet_RL/ns-3.46"
    output_dir = "vanet-dqn-eval"
    env = NS3Environment(ns3_path, output_dir)
    
    # Initialize and load agent
    state_size = 8
    action_size = 5
    agent = DQNAgent(state_size, action_size)
    agent.load_model(model_path)
    
    # Evaluation statistics
    eval_rewards = []
    eval_throughputs = []
    eval_delays = []
    eval_pdrs = []
    
    for episode in range(num_episodes):
        print(f"\nEvaluation Episode {episode + 1}/{num_episodes}")
        
        # Run episode without training
        episode_reward, steps = env.run_episode(agent, training=False)
        
        if episode_reward is not None:
            eval_rewards.append(episode_reward)
            
            stats = env.get_flow_stats()
            if stats:
                eval_throughputs.append(stats.get('throughput', 0))
                eval_delays.append(stats.get('delay', 0))
                eval_pdrs.append(stats.get('pdr', 0))
                
                print(f"Throughput: {stats.get('throughput', 0):.2f} kbps")
                print(f"Delay: {stats.get('delay', 0):.4f} s")
                print(f"PDR: {stats.get('pdr', 0):.2f}%")
    
    # Print summary
    print(f"\n{'='*70}")
    print("Evaluation Summary:")
    print(f"Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Average Throughput: {np.mean(eval_throughputs):.2f} ± {np.std(eval_throughputs):.2f} kbps")
    print(f"Average Delay: {np.mean(eval_delays):.4f} ± {np.std(eval_delays):.4f} s")
    print(f"Average PDR: {np.mean(eval_pdrs):.2f} ± {np.std(eval_pdrs):.2f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train VANET DQN Agent')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                       help='Mode: train or eval')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_dqn_agent(episodes=args.episodes)
    elif args.mode == 'eval':
        if args.model is None:
            print("Error: --model path required for evaluation")
            sys.exit(1)
        evaluate_agent(args.model)
