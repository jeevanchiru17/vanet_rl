#!/bin/bash

# VANET DQN Setup Script
# This script sets up the environment and builds NS-3

echo "======================================================================"
echo "VANET Traffic Congestion Control - Setup Script"
echo "======================================================================"
echo ""

# Step 1: Check Python version
echo "Step 1: Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi
echo "✓ Python 3 is available"
echo ""

# Step 2: Install Python dependencies
echo "Step 2: Installing Python dependencies..."
echo "This may take a few minutes..."
pip3 install --upgrade pip
pip3 install torch numpy matplotlib pandas gym --quiet
if [ $? -eq 0 ]; then
    echo "✓ Python dependencies installed successfully"
else
    echo "! Warning: Some dependencies may have failed to install"
fi
echo ""

# Step 3: Configure NS-3
echo "Step 3: Configuring NS-3..."
cd ns-3.46
./ns3 configure --enable-examples --enable-tests
if [ $? -ne 0 ]; then
    echo "Error: NS-3 configuration failed"
    exit 1
fi
echo "✓ NS-3 configured successfully"
echo ""

# Step 4: Build NS-3
echo "Step 4: Building NS-3..."
echo "This will take several minutes..."
./ns3 build
if [ $? -ne 0 ]; then
    echo "Error: NS-3 build failed"
    exit 1
fi
echo "✓ NS-3 built successfully"
echo ""

# Step 5: Verify the simulation file
echo "Step 5: Verifying simulation file..."
if [ -f "scratch/vanet-dqn-simulation.cc" ]; then
    echo "✓ VANET DQN simulation file found"
else
    echo "Error: Simulation file not found"
    exit 1
fi
echo ""

cd ..

# Step 6: Create output directories
echo "Step 6: Creating output directories..."
mkdir -p vanet-dqn-results
mkdir -p training_results
echo "✓ Output directories created"
echo ""

echo "======================================================================"
echo "✓✓✓ Setup completed successfully! ✓✓✓"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Test the DQN agent: python3 test_dqn.py"
echo "  2. Run a single NS-3 simulation:"
echo "     cd ns-3.46"
echo "     ./ns3 run 'scratch/vanet-dqn-simulation --nVehicles=50 --simTime=100'"
echo "  3. Start training: python3 train_vanet_dqn.py --mode train --episodes 10"
echo ""
echo "======================================================================"
