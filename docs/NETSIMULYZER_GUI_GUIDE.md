# 🎨 NetSimulyzer GUI - User Guide

## 🎯 Overview

Professional web-based GUI for visualizing NS-3 NetSimulyzer simulation data. Perfect for VANET RL project visualization!

**File**: `netsimulyzer-gui.html` ✅ **NOW OPEN!**

---

## ✨ Features

### 📊 **Complete Visualization Suite**
- **3D Scene**: Interactive Three.js visualization
- **Playback Controls**: Play, pause, reset, speed control
- **Timeline**: Scrub through simulation time
- **Statistics**: Real-time metrics display
- **Node List**: Browse and focus on individual nodes
- **Display Options**: Toggle labels, trails, grid, connections

### 🎮 **Interactive Controls**
- **Camera**: Drag to rotate, scroll to zoom
- **Timeline**: Scrub to any point in simulation
- **Speed**: 0.1x to 5x playback speed
- **Nodes**: Click to focus on specific vehicles

---

## 🚀 How to Use

### Step 1: Open the GUI

```bash
cd /Users/jeevanhr/vanet_RL
open netsimulyzer-gui.html
```

### Step 2: Load Simulation Data

1. Click **"📁 Load Simulation"** button in sidebar
2. Navigate to: `ns-3.46/vanet-test.json`
3. Select the file
4. Wait for loading (shows statistics when ready)

### Step 3: Explore

**Playback**:
- Click **▶️ Play** to start animation
- Click **⏸️ Pause** to stop
- Click **🔄 Reset** to restart
- Adjust **Speed** slider (0.1x - 5x)

**View**:
- **Drag mouse** to rotate camera
- **Scroll wheel** to zoom in/out
- **Timeline slider** to jump to specific time

**Display Options**:
- ✅ **Show Node Labels**: Vehicle IDs
- ✅ **Show Motion Trails**: Path history
- ✅ **Show Grid**: Ground reference
- ☐ **Show Connections**: V2V links

**Node List**:
- Click any node to focus camera on it
- See all vehicles in simulation

---

## 📊 GUI Layout

```
┌─────────────────────────────────────────────────────┐
│  Sidebar (Left)         │  Main Canvas (Right)      │
│                         │                           │
│  📁 Load Simulation     │  🎨 3D Visualization      │
│  📊 Statistics          │                           │
│  ⏯️  Playback Controls  │  ℹ️  Info Overlay         │
│  🎨 Display Options     │                           │
│  🔍 Node List           │  ⏱️  Timeline (Bottom)    │
└─────────────────────────────────────────────────────┘
```

---

## 🎨 Sidebar Sections

### 1. **Load Simulation**
- File picker for JSON files
- Shows loaded filename
- Displays loading spinner

### 2. **Statistics**
- **Nodes**: Total vehicle count
- **Duration**: Simulation length
- **Events**: Number of events
- **Series**: Data series count

### 3. **Playback Controls**
- **Play/Pause/Reset** buttons
- **Speed slider**: Control playback rate
- Shows current speed (e.g., "1.0x")

### 4. **Display Options**
- **Node Labels**: Show/hide vehicle IDs
- **Motion Trails**: Show/hide paths
- **Grid**: Show/hide ground grid
- **Connections**: Show/hide V2V links

### 5. **Node List**
- Scrollable list of all vehicles
- Click to focus camera
- Highlights selected node

---

## 🖥️ Main Canvas

### Info Overlay (Top Right)
Shows real-time information:
- **Time**: Current simulation time
- **Active Nodes**: Number of visible vehicles
- **Camera**: Control hints
- **Zoom**: Control hints

### Timeline (Bottom)
- **Slider**: Scrub through time
- **Start Time**: 0.0s
- **Current Time**: Updates in real-time
- **End Time**: Total duration

---

## 🎯 Use Cases

### 1. **Debugging NS-3 Simulations**
- Load simulation output
- Check vehicle positions
- Verify movement patterns
- Inspect timing

### 2. **Presentation**
- Professional visualization
- Interactive demo
- Real-time playback
- Multiple viewing angles

### 3. **Analysis**
- Study vehicle behavior
- Analyze traffic patterns
- Compare scenarios
- Extract insights

### 4. **RL Training Visualization**
- Visualize agent decisions
- See traffic flow changes
- Compare before/after RL
- Validate improvements

---

## 🔧 Integration with RL Project

### Workflow

```bash
# 1. Train RL agent
python train_vanet_dqn.py --mode train --episodes 100

# 2. Run NS-3 simulation with trained agent
cd ns-3.46
./ns3 run vanet-netsimulyzer-test

# 3. Visualize results in GUI
cd ..
open netsimulyzer-gui.html
# Load: ns-3.46/vanet-test.json
```

### Compare Scenarios

**Baseline (No RL)**:
```bash
./ns3 run "vanet-netsimulyzer-test --rl=false"
# Save as: baseline.json
```

**With RL**:
```bash
./ns3 run "vanet-netsimulyzer-test --rl=true"
# Save as: with-rl.json
```

**Visualize Both**:
1. Load `baseline.json` → observe behavior
2. Load `with-rl.json` → compare improvement

---

## 💡 Tips

### For Best Visualization

1. **Camera Position**
   - Start with top-down view
   - Rotate to see 3D perspective
   - Zoom to appropriate level

2. **Playback Speed**
   - Use 1x for normal viewing
   - Use 0.5x for detailed analysis
   - Use 2-3x for overview

3. **Display Options**
   - Enable trails to see paths
   - Enable labels for identification
   - Disable grid for cleaner view

4. **Timeline**
   - Scrub to interesting moments
   - Pause to examine details
   - Reset to start over

### For Presentations

1. **Preparation**
   - Load file before presenting
   - Set camera to good angle
   - Choose display options
   - Test playback speed

2. **During Demo**
   - Start paused, explain layout
   - Play at 1x speed
   - Rotate camera to show angles
   - Pause to highlight points
   - Use timeline to jump to events

3. **Interactive**
   - Let audience suggest camera angles
   - Focus on specific vehicles
   - Adjust speed as needed
   - Answer questions with visualization

---

## 🎨 Customization

### Colors

Edit the HTML file to change colors:

```javascript
// Vehicle color
color: 0x3b82f6  // Bright blue

// Background
scene.background = new THREE.Color(0x1a1a2e);

// Grid
const gridHelper = new THREE.GridHelper(1000, 50, 0x667eea, 0x444444);
```

### Camera

Adjust initial camera position:

```javascript
camera.position.set(0, 500, 500);  // x, y, z
camera.lookAt(0, 0, 0);
```

### Node Appearance

Change vehicle size/shape:

```javascript
const geometry = new THREE.BoxGeometry(10, 5, 15);  // width, height, depth
```

---

## 📊 Comparison with Other Visualizers

| Feature | NetSimulyzer GUI | Roundabout 3D | NetSimulyzer App |
|---------|------------------|---------------|------------------|
| **Load JSON** | ✅ Yes | ❌ No | ✅ Yes |
| **3D View** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Playback** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Node List** | ✅ Yes | ❌ No | ✅ Yes |
| **Statistics** | ✅ Yes | ✅ Yes | ✅ Yes |
| **macOS Build** | ✅ Works | ✅ Works | ❌ Fails |
| **Auto-sim** | ❌ No | ✅ Yes | ❌ No |
| **Data Source** | NS-3 JSON | Built-in | NS-3 JSON |

**Use NetSimulyzer GUI when**: You have NS-3 JSON output to visualize
**Use Roundabout 3D when**: You want auto-running demo with traffic rules

---

## 🐛 Troubleshooting

### File Won't Load
- Check file is valid JSON
- Ensure it's NetSimulyzer format
- Try smaller file first

### No Nodes Visible
- Check statistics show nodes > 0
- Zoom out (scroll wheel)
- Reset camera position

### Playback Not Working
- Click Play button
- Check speed is not 0
- Verify file loaded correctly

### Performance Issues
- Reduce number of nodes
- Disable motion trails
- Lower playback speed

---

## 🌟 Summary

**NetSimulyzer GUI** provides:

✅ **Professional visualization** of NS-3 data
✅ **Interactive controls** for exploration
✅ **Real-time playback** with timeline
✅ **Statistics display** for analysis
✅ **Node management** for focus
✅ **Works on macOS** (no build issues!)

**Perfect for**:
- ✅ Visualizing RL training results
- ✅ Debugging NS-3 simulations
- ✅ Project presentations
- ✅ Traffic analysis

---

## 📁 Files

**Main GUI**: `netsimulyzer-gui.html`
**Data Source**: `ns-3.46/vanet-test.json`
**Alternative**: `vanet-roundabout-3d.html` (auto-running demo)

---

**Status**: ✅ **Ready to use!**
**Open**: `open netsimulyzer-gui.html`
**Load**: `ns-3.46/vanet-test.json`

**Perfect for your VANET RL project visualization!** 🎯
