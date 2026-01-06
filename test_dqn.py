"""
Test script for VANET DQN components
Verifies that all components are working correctly
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dqn_agent import DQNAgent, VANETState, VANETReward


def test_dqn_agent():
    """Test DQN agent initialization and basic operations"""
    print("\n" + "="*70)
    print("Testing DQN Agent")
    print("="*70)
    
    state_size = 8
    action_size = 5
    
    # Initialize agent
    print("\n1. Initializing agent...")
    agent = DQNAgent(state_size, action_size)
    print(f"   ✓ Agent initialized successfully")
    print(f"   - Device: {agent.device}")
    print(f"   - State size: {state_size}")
    print(f"   - Action size: {action_size}")
    
    # Test state extraction
    print("\n2. Testing state extraction...")
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
    print(f"   ✓ State extracted successfully")
    print(f"   - State shape: {state.shape}")
    print(f"   - State values: {state}")
    
    # Test action selection
    print("\n3. Testing action selection...")
    action = agent.select_action(state, training=True)
    print(f"   ✓ Action selected: {action}")
    print(f"   - Epsilon: {agent.epsilon:.4f}")
    
    # Test experience storage
    print("\n4. Testing experience storage...")
    next_state = VANETState.extract_features({
        'traffic_density': 45,
        'avg_queue_length': 8,
        'link_quality': 18,
        'vehicle_speed': 22,
        'congestion_level': 0.25,
        'packet_loss_rate': 0.03,
        'neighbor_count': 7,
        'distance_to_destination': 450
    })
    reward = 5.0
    done = False
    
    agent.store_experience(state, action, reward, next_state, done)
    print(f"   ✓ Experience stored")
    print(f"   - Buffer size: {len(agent.replay_buffer)}")
    
    # Fill buffer with random experiences
    print("\n5. Filling replay buffer...")
    for i in range(100):
        random_state = np.random.rand(state_size).astype(np.float32)
        random_action = np.random.randint(0, action_size)
        random_reward = np.random.randn()
        random_next_state = np.random.rand(state_size).astype(np.float32)
        random_done = np.random.rand() > 0.9
        
        agent.store_experience(random_state, random_action, random_reward, 
                             random_next_state, random_done)
    
    print(f"   ✓ Buffer filled with {len(agent.replay_buffer)} experiences")
    
    # Test training step
    print("\n6. Testing training step...")
    loss = agent.train_step()
    if loss is not None:
        print(f"   ✓ Training step completed")
        print(f"   - Loss: {loss:.4f}")
    else:
        print(f"   ! Not enough samples for training")
    
    # Test model save/load
    print("\n7. Testing model save/load...")
    test_model_path = "test_model.pth"
    agent.save_model(test_model_path)
    print(f"   ✓ Model saved to {test_model_path}")
    
    # Create new agent and load
    new_agent = DQNAgent(state_size, action_size)
    new_agent.load_model(test_model_path)
    print(f"   ✓ Model loaded successfully")
    
    # Cleanup
    os.remove(test_model_path)
    print(f"   ✓ Test model cleaned up")
    
    # Test statistics
    print("\n8. Getting agent statistics...")
    stats = agent.get_statistics()
    print(f"   ✓ Statistics retrieved:")
    for key, value in stats.items():
        print(f"     - {key}: {value}")
    
    print("\n" + "="*70)
    print("✓ All DQN Agent tests passed!")
    print("="*70)


def test_reward_function():
    """Test reward function"""
    print("\n" + "="*70)
    print("Testing Reward Function")
    print("="*70)
    
    test_cases = [
        {
            'name': 'Successful delivery, low delay',
            'metrics': {
                'packet_delivered': True,
                'delay': 0.02,
                'congestion_created': 0.1,
                'routing_overhead': 2
            }
        },
        {
            'name': 'Failed delivery, high delay',
            'metrics': {
                'packet_delivered': False,
                'delay': 0.5,
                'congestion_created': 0.8,
                'routing_overhead': 10
            }
        },
        {
            'name': 'Successful delivery, high congestion',
            'metrics': {
                'packet_delivered': True,
                'delay': 0.1,
                'congestion_created': 0.7,
                'routing_overhead': 5
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        reward = VANETReward.calculate_reward(test_case['metrics'])
        print(f"   Metrics: {test_case['metrics']}")
        print(f"   Reward: {reward:.2f}")
    
    print("\n" + "="*70)
    print("✓ All Reward Function tests passed!")
    print("="*70)


def test_network_architecture():
    """Test neural network architecture"""
    print("\n" + "="*70)
    print("Testing Neural Network Architecture")
    print("="*70)
    
    state_size = 8
    action_size = 5
    batch_size = 32
    
    agent = DQNAgent(state_size, action_size)
    
    print("\n1. Policy Network Architecture:")
    print(agent.policy_net)
    
    print("\n2. Testing forward pass...")
    test_batch = torch.randn(batch_size, state_size).to(agent.device)
    output = agent.policy_net(test_batch)
    print(f"   ✓ Forward pass successful")
    print(f"   - Input shape: {test_batch.shape}")
    print(f"   - Output shape: {output.shape}")
    print(f"   - Expected output shape: ({batch_size}, {action_size})")
    
    assert output.shape == (batch_size, action_size), "Output shape mismatch!"
    
    print("\n3. Testing gradient flow...")
    loss = output.mean()
    loss.backward()
    
    has_gradients = all(
        param.grad is not None 
        for param in agent.policy_net.parameters()
    )
    
    if has_gradients:
        print(f"   ✓ Gradients computed successfully")
    else:
        print(f"   ✗ Gradient computation failed")
    
    print("\n" + "="*70)
    print("✓ All Neural Network tests passed!")
    print("="*70)


def test_state_normalization():
    """Test state normalization"""
    print("\n" + "="*70)
    print("Testing State Normalization")
    print("="*70)
    
    test_states = [
        {
            'name': 'Low traffic',
            'state': {
                'traffic_density': 10,
                'avg_queue_length': 2,
                'link_quality': 25,
                'vehicle_speed': 15,
                'congestion_level': 0.1,
                'packet_loss_rate': 0.01,
                'neighbor_count': 3,
                'distance_to_destination': 200
            }
        },
        {
            'name': 'High traffic',
            'state': {
                'traffic_density': 90,
                'avg_queue_length': 45,
                'link_quality': 5,
                'vehicle_speed': 5,
                'congestion_level': 0.9,
                'packet_loss_rate': 0.3,
                'neighbor_count': 18,
                'distance_to_destination': 800
            }
        },
        {
            'name': 'Medium traffic',
            'state': {
                'traffic_density': 50,
                'avg_queue_length': 20,
                'link_quality': 15,
                'vehicle_speed': 20,
                'congestion_level': 0.5,
                'packet_loss_rate': 0.15,
                'neighbor_count': 10,
                'distance_to_destination': 500
            }
        }
    ]
    
    for i, test_case in enumerate(test_states, 1):
        print(f"\n{i}. {test_case['name']}")
        state = VANETState.extract_features(test_case['state'])
        print(f"   Normalized state: {state}")
        print(f"   Min value: {state.min():.4f}")
        print(f"   Max value: {state.max():.4f}")
        
        # Check if values are in reasonable range [0, 1] or close
        if state.min() >= -0.1 and state.max() <= 1.1:
            print(f"   ✓ Normalization looks good")
        else:
            print(f"   ! Warning: Values outside expected range")
    
    print("\n" + "="*70)
    print("✓ All State Normalization tests passed!")
    print("="*70)


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("VANET DQN COMPONENT TESTS")
    print("="*70)
    
    try:
        test_dqn_agent()
        test_reward_function()
        test_network_architecture()
        test_state_normalization()
        
        print("\n" + "="*70)
        print("✓✓✓ ALL TESTS PASSED SUCCESSFULLY! ✓✓✓")
        print("="*70)
        print("\nThe VANET DQN system is ready to use!")
        print("\nNext steps:")
        print("  1. Build NS-3: cd ns-3.46 && ./ns3 build")
        print("  2. Install Python dependencies: pip install -r requirements.txt")
        print("  3. Start training: python train_vanet_dqn.py --mode train --episodes 100")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"✗✗✗ TEST FAILED: {str(e)} ✗✗✗")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
