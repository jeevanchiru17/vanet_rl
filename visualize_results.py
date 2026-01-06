"""
Visualization tools for VANET DQN results
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.animation import FuncAnimation
import pandas as pd


def plot_training_comparison(history_files, labels, output_file='comparison.png'):
    """
    Compare multiple training runs
    
    Args:
        history_files: List of paths to training_history.json files
        labels: List of labels for each run
        output_file: Output filename
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('VANET DQN Training Comparison', fontsize=18, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(history_files)))
    
    for idx, (history_file, label) in enumerate(zip(history_files, labels)):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        color = colors[idx]
        
        # Episode rewards
        rewards = history['episode_rewards']
        axes[0, 0].plot(rewards, label=label, color=color, alpha=0.4)
        if len(rewards) > 10:
            moving_avg = pd.Series(rewards).rolling(window=10).mean()
            axes[0, 0].plot(moving_avg, color=color, linewidth=2)
        
        # Throughput
        if 'throughputs' in history and history['throughputs']:
            throughputs = history['throughputs']
            axes[0, 1].plot(throughputs, label=label, color=color, alpha=0.4)
            if len(throughputs) > 10:
                moving_avg = pd.Series(throughputs).rolling(window=10).mean()
                axes[0, 1].plot(moving_avg, color=color, linewidth=2)
        
        # Delay
        if 'delays' in history and history['delays']:
            delays = history['delays']
            axes[1, 0].plot(delays, label=label, color=color, alpha=0.4)
            if len(delays) > 10:
                moving_avg = pd.Series(delays).rolling(window=10).mean()
                axes[1, 0].plot(moving_avg, color=color, linewidth=2)
        
        # PDR
        if 'pdrs' in history and history['pdrs']:
            pdrs = history['pdrs']
            axes[1, 1].plot(pdrs, label=label, color=color, alpha=0.4)
            if len(pdrs) > 10:
                moving_avg = pd.Series(pdrs).rolling(window=10).mean()
                axes[1, 1].plot(moving_avg, color=color, linewidth=2)
    
    # Configure subplots
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Total Reward', fontsize=12)
    axes[0, 0].set_title('Episode Rewards', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Throughput (kbps)', fontsize=12)
    axes[0, 1].set_title('Network Throughput', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Episode', fontsize=12)
    axes[1, 0].set_ylabel('Delay (s)', fontsize=12)
    axes[1, 0].set_title('End-to-End Delay', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Episode', fontsize=12)
    axes[1, 1].set_ylabel('PDR (%)', fontsize=12)
    axes[1, 1].set_title('Packet Delivery Ratio', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")
    plt.show()


def plot_q_values_heatmap(agent, state_samples, output_file='q_values_heatmap.png'):
    """
    Visualize Q-values for different states
    
    Args:
        agent: Trained DQN agent
        state_samples: List of state vectors
        output_file: Output filename
    """
    import torch
    
    q_values_list = []
    
    for state in state_samples:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
            q_values_list.append(q_values)
    
    q_values_array = np.array(q_values_list)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(q_values_array.T, cmap='RdYlGn', aspect='auto')
    
    ax.set_xlabel('State Sample', fontsize=12)
    ax.set_ylabel('Action', fontsize=12)
    ax.set_title('Q-Values Heatmap', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Q-Value', fontsize=12)
    
    # Set ticks
    ax.set_yticks(np.arange(q_values_array.shape[1]))
    ax.set_yticklabels([f'Action {i}' for i in range(q_values_array.shape[1])])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Q-values heatmap saved to {output_file}")
    plt.show()


def plot_network_metrics_distribution(stats_file, output_file='metrics_distribution.png'):
    """
    Plot distribution of network metrics
    
    Args:
        stats_file: Path to flow-stats.txt
        output_file: Output filename
    """
    # Parse flow statistics
    flows = []
    
    with open(stats_file, 'r') as f:
        content = f.read()
        flow_blocks = content.split('Flow ')[1:]  # Skip header
        
        for block in flow_blocks:
            if 'Overall Statistics' in block:
                break
            
            flow_data = {}
            for line in block.split('\n'):
                if 'Tx Packets:' in line:
                    flow_data['tx_packets'] = int(line.split(':')[1].strip())
                elif 'Rx Packets:' in line:
                    flow_data['rx_packets'] = int(line.split(':')[1].strip())
                elif 'Throughput:' in line:
                    flow_data['throughput'] = float(line.split(':')[1].strip().split()[0])
                elif 'Mean Delay:' in line:
                    flow_data['delay'] = float(line.split(':')[1].strip().split()[0])
                elif 'Packet Loss:' in line:
                    flow_data['loss'] = int(line.split(':')[1].strip())
            
            if flow_data:
                flows.append(flow_data)
    
    if not flows:
        print("No flow data found")
        return
    
    # Create distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Network Metrics Distribution', fontsize=16, fontweight='bold')
    
    # Throughput distribution
    throughputs = [f['throughput'] for f in flows if 'throughput' in f]
    axes[0, 0].hist(throughputs, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(throughputs), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(throughputs):.2f}')
    axes[0, 0].set_xlabel('Throughput (kbps)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Throughput Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Delay distribution
    delays = [f['delay'] for f in flows if 'delay' in f and f['delay'] > 0]
    axes[0, 1].hist(delays, bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(delays), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(delays):.4f}')
    axes[0, 1].set_xlabel('Delay (s)', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Delay Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Packet delivery
    tx_packets = [f['tx_packets'] for f in flows if 'tx_packets' in f]
    rx_packets = [f['rx_packets'] for f in flows if 'rx_packets' in f]
    
    x = np.arange(len(tx_packets))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, tx_packets, width, label='Transmitted', 
                   color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, rx_packets, width, label='Received', 
                   color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Flow ID', fontsize=11)
    axes[1, 0].set_ylabel('Packets', fontsize=11)
    axes[1, 0].set_title('Packet Transmission vs Reception', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Packet loss
    losses = [f['loss'] for f in flows if 'loss' in f]
    axes[1, 1].bar(range(len(losses)), losses, color='red', alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Flow ID', fontsize=11)
    axes[1, 1].set_ylabel('Lost Packets', fontsize=11)
    axes[1, 1].set_title('Packet Loss per Flow', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Metrics distribution plot saved to {output_file}")
    plt.show()


def create_performance_report(history_file, output_file='performance_report.txt'):
    """
    Generate a detailed performance report
    
    Args:
        history_file: Path to training_history.json
        output_file: Output filename
    """
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    rewards = history['episode_rewards']
    throughputs = history.get('throughputs', [])
    delays = history.get('delays', [])
    pdrs = history.get('pdrs', [])
    
    report = []
    report.append("=" * 70)
    report.append("VANET DQN PERFORMANCE REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Training configuration
    report.append("TRAINING CONFIGURATION")
    report.append("-" * 70)
    config = history.get('config', {})
    for key, value in config.items():
        report.append(f"  {key}: {value}")
    report.append("")
    
    # Episode statistics
    report.append("EPISODE STATISTICS")
    report.append("-" * 70)
    report.append(f"  Total Episodes: {len(rewards)}")
    report.append(f"  Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    report.append(f"  Best Reward: {np.max(rewards):.2f} (Episode {np.argmax(rewards) + 1})")
    report.append(f"  Worst Reward: {np.min(rewards):.2f} (Episode {np.argmin(rewards) + 1})")
    report.append("")
    
    # Last 10 episodes performance
    if len(rewards) >= 10:
        report.append("LAST 10 EPISODES PERFORMANCE")
        report.append("-" * 70)
        report.append(f"  Average Reward: {np.mean(rewards[-10:]):.2f}")
        if throughputs:
            report.append(f"  Average Throughput: {np.mean(throughputs[-10:]):.2f} kbps")
        if delays:
            report.append(f"  Average Delay: {np.mean(delays[-10:]):.4f} s")
        if pdrs:
            report.append(f"  Average PDR: {np.mean(pdrs[-10:]):.2f}%")
        report.append("")
    
    # Network performance
    if throughputs:
        report.append("NETWORK PERFORMANCE")
        report.append("-" * 70)
        report.append(f"  Throughput:")
        report.append(f"    Mean: {np.mean(throughputs):.2f} kbps")
        report.append(f"    Std: {np.std(throughputs):.2f} kbps")
        report.append(f"    Max: {np.max(throughputs):.2f} kbps")
        report.append(f"    Min: {np.min(throughputs):.2f} kbps")
        report.append("")
    
    if delays:
        report.append(f"  Delay:")
        report.append(f"    Mean: {np.mean(delays):.4f} s")
        report.append(f"    Std: {np.std(delays):.4f} s")
        report.append(f"    Max: {np.max(delays):.4f} s")
        report.append(f"    Min: {np.min(delays):.4f} s")
        report.append("")
    
    if pdrs:
        report.append(f"  Packet Delivery Ratio:")
        report.append(f"    Mean: {np.mean(pdrs):.2f}%")
        report.append(f"    Std: {np.std(pdrs):.2f}%")
        report.append(f"    Max: {np.max(pdrs):.2f}%")
        report.append(f"    Min: {np.min(pdrs):.2f}%")
        report.append("")
    
    # Improvement analysis
    if len(rewards) >= 20:
        report.append("IMPROVEMENT ANALYSIS")
        report.append("-" * 70)
        first_10_avg = np.mean(rewards[:10])
        last_10_avg = np.mean(rewards[-10:])
        improvement = ((last_10_avg - first_10_avg) / abs(first_10_avg)) * 100
        report.append(f"  First 10 episodes avg reward: {first_10_avg:.2f}")
        report.append(f"  Last 10 episodes avg reward: {last_10_avg:.2f}")
        report.append(f"  Improvement: {improvement:+.2f}%")
        report.append("")
    
    report.append("=" * 70)
    
    # Write report
    report_text = '\n'.join(report)
    with open(output_file, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\nReport saved to {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize VANET DQN results')
    parser.add_argument('--history', type=str, help='Path to training_history.json')
    parser.add_argument('--stats', type=str, help='Path to flow-stats.txt')
    parser.add_argument('--compare', nargs='+', help='Paths to multiple history files for comparison')
    parser.add_argument('--labels', nargs='+', help='Labels for comparison')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    
    args = parser.parse_args()
    
    if args.compare and args.labels:
        if len(args.compare) != len(args.labels):
            print("Error: Number of history files must match number of labels")
        else:
            plot_training_comparison(args.compare, args.labels)
    
    if args.stats:
        plot_network_metrics_distribution(args.stats)
    
    if args.history and args.report:
        create_performance_report(args.history)
    
    if not any([args.compare, args.stats, args.history]):
        print("No visualization specified. Use --help for options.")
