# VANET Traffic Congestion Control using Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NS-3](https://img.shields.io/badge/NS--3-3.46-blue.svg)](https://www.nsnam.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Deep Q-Network (DQN) based intelligent traffic congestion control for Vehicular Ad-hoc Networks (VANETs) using NS-3 simulation.

## 🎯 Project Overview

This project implements a **Reinforcement Learning** approach to solve traffic congestion in VANETs by making intelligent routing decisions. The system uses:

- **NS-3 Network Simulator** for realistic VANET simulation
- **Deep Q-Network (DQN)** for learning optimal routing policies
- **IEEE 802.11p (WAVE)** for vehicle-to-vehicle communication
- **3D Visualization** for interactive demonstration

### Key Features

- ✅ **Realistic VANET Simulation** with IEEE 802.11p
- ✅ **DQN Agent** with experience replay and target network
- ✅ **Traffic Congestion Scenarios** with multiple vehicles
- ✅ **Performance Metrics** (PDR, throughput, delay)
- ✅ **3D Visualizers** for presentation and analysis
- ✅ **NetSimulyzer Integration** for detailed visualization

## 🚀 Quick Start

### Prerequisites

- **NS-3.46** (included in `ns-3.46/`)
- **Python 3.8+**
- **PyTorch 2.0+**
- **macOS/Linux** (tested on macOS)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/vanet_RL.git
cd vanet_RL

# Install Python dependencies
pip install -r requirements.txt

# Build NS-3
cd ns-3.46
./ns3 configure --enable-examples
./ns3 build
cd ..
```

### Run Training

```bash
# Train DQN agent (100 episodes)
python train_vanet_dqn.py --mode train --episodes 100

# Evaluate trained model
python train_vanet_dqn.py --mode eval --model vanet-dqn-results/best_model.pth
```

### Visualize Results

```bash
# Generate training plots
python visualize_results.py

# Open 3D roundabout visualizer
open vanet-roundabout-3d.html

# Open NetSimulyzer GUI
open netsimulyzer-gui.html
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Training Loop                         │
│                                                          │
│  Python (DQN Agent)  ←→  NS-3 (VANET Simulation)       │
│                                                          │
│  1. Agent selects action (routing decision)             │
│  2. NS-3 simulates network with that action             │
│  3. NS-3 returns state & reward                         │
│  4. Agent learns from experience                        │
│  5. Repeat                                              │
└─────────────────────────────────────────────────────────┘
```

### State Space (7 features)
- Traffic Density (vehicles/km)
- Average Queue Length (packets)
- Link Quality (SINR in dB)
- Vehicle Speed (m/s)
- Congestion Level (0-1)
- Packet Loss Rate (0-1)
- Neighbor Count

### Action Space (4 actions)
- Route 0: Direct path
- Route 1: Alternative path 1
- Route 2: Alternative path 2
- Route 3: Adaptive routing

### Reward Function
```python
reward = (
    +10.0  if packet delivered
    -5.0   if packet dropped
    -delay * 2.0
    -congestion * 3.0
    -routing_overhead * 0.1
)
```

## 📁 Project Structure

```
vanet_RL/
├── dqn_agent.py                    # DQN implementation
├── train_vanet_dqn.py              # Training script
├── test_dqn.py                     # Testing script
├── visualize_results.py            # Results visualization
├── requirements.txt                # Python dependencies
├── vanet-roundabout-3d.html       # 3D visualizer
├── netsimulyzer-gui.html          # NetSimulyzer GUI
├── ns-3.46/
│   └── scratch/
│       └── vanet-dqn-simulation.cc # NS-3 simulation
├── docs/
│   ├── RL_IMPLEMENTATION_GUIDE.md
│   ├── VANET_VISUALIZER_GUIDE.md
│   └── NETSIMULYZER_GUI_GUIDE.md
└── README.md
```

## 🧠 DQN Agent

### Network Architecture
- **Input**: 7-dimensional state vector
- **Hidden Layers**: [256, 128] neurons with ReLU
- **Output**: 4 Q-values (one per action)

### Hyperparameters
```python
learning_rate = 0.0001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
batch_size = 64
replay_buffer_size = 100000
target_update_frequency = 10 episodes
```

## 🎨 Visualization Tools

### 1. 3D Roundabout Visualizer
Interactive 3D simulation with:
- Multi-lane roundabout
- 15 vehicles with realistic models
- Traffic rules (yielding, safe distance, speed limits)
- V2V communication links
- Motion trails

**Usage**: `open vanet-roundabout-3d.html`

### 2. NetSimulyzer GUI
Professional web-based GUI for NS-3 output:
- Load NetSimulyzer JSON files
- 3D visualization with playback controls
- Real-time statistics
- Node list and focus
- Timeline scrubbing

**Usage**: `open netsimulyzer-gui.html`

### 3. Training Plots
Matplotlib-based visualization:
- Reward curves
- Performance metrics (PDR, throughput, delay)
- Comparison with baseline

**Usage**: `python visualize_results.py`

## 📈 Results

### Expected Performance

| Metric | Baseline (No RL) | With DQN |
|--------|------------------|----------|
| **PDR** | 70-80% | **90-95%** ✅ |
| **Avg Delay** | 150-200ms | **80-120ms** ✅ |
| **Throughput** | 500 kbps | **700-900 kbps** ✅ |
| **Congestion** | High | **Low** ✅ |

### Training Convergence
- **Episodes 1-20**: Exploration phase
- **Episodes 20-50**: Learning phase
- **Episodes 50-80**: Convergence
- **Episodes 80-100**: Fine-tuning

## 🔧 Configuration

### NS-3 Simulation Parameters

```bash
./ns3 run "vanet-dqn-simulation \
  --nVehicles=50 \
  --simTime=100 \
  --areaSize=1000 \
  --packetSize=1024 \
  --packetInterval=0.1"
```

### DQN Training Parameters

Edit `dqn_agent.py` to modify:
- Network architecture
- Learning rate
- Exploration strategy
- Replay buffer size

## 📚 Documentation

- **[RL Implementation Guide](docs/RL_IMPLEMENTATION_GUIDE.md)**: Complete setup and training guide
- **[Visualizer Guide](docs/VANET_VISUALIZER_GUIDE.md)**: 3D visualizer usage
- **[NetSimulyzer GUI Guide](docs/NETSIMULYZER_GUI_GUIDE.md)**: GUI documentation

## 🧪 Testing

### Run Tests

```bash
# Test DQN agent
python test_dqn.py

# Test NS-3 simulation
cd ns-3.46
./ns3 run vanet-dqn-simulation
cd ..

# Integration test
python -c "from dqn_agent import DQNAgent; agent = DQNAgent(7, 4); print('✅ OK')"
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NS-3** - Network Simulator
- **PyTorch** - Deep Learning Framework
- **NetSimulyzer** - NS-3 Visualization Tool
- **Three.js** - 3D Graphics Library

## 📧 Contact

**Project Link**: [https://github.com/jeevanchiru17/vanet_RL](https://github.com/jeevanchiru17/vanet_RL)

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@misc{vanet_rl_2026,
  title={VANET Traffic Congestion Control using Deep Reinforcement Learning},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  url={https://github.com/YOUR_USERNAME/vanet_RL}
}
```

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Made with ❤️ for Advanced Computer Networks**
