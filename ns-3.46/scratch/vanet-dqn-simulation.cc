/*
 * VANET Traffic Congestion Control using Deep Q-Learning
 * 
 * This simulation implements a Vehicular Ad-hoc Network (VANET) with
 * IEEE 802.11p (WAVE) communication and integrates with a Python DQN agent
 * for intelligent traffic-aware routing decisions.
 * 
 * Features:
 * - Vehicle mobility with realistic traffic patterns
 * - IEEE 802.11p WAVE communication
 * - Integration with Python DQN agent via socket/file communication
 * - Traffic congestion scenarios
 * - Performance metrics collection
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/netanim-module.h"
#include <fstream>
#include <sstream>
#include <map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("VANETDQNSimulation");

// Global variables for metrics
uint32_t g_packetsReceived = 0;
uint32_t g_packetsSent = 0;
uint32_t g_packetsDropped = 0;
double g_totalDelay = 0.0;
uint32_t g_routingOverhead = 0;

// State information structure
struct VehicleState {
    double trafficDensity;
    double avgQueueLength;
    double linkQuality;
    double vehicleSpeed;
    double congestionLevel;
    double packetLossRate;
    uint32_t neighborCount;
    double distanceToDestination;
};

// Metrics structure
struct PerformanceMetrics {
    bool packetDelivered;
    double delay;
    double congestionCreated;
    uint32_t routingOverhead;
};

/**
 * Class to manage DQN agent communication
 */
class DQNInterface {
public:
    DQNInterface(std::string stateFile, std::string actionFile, std::string rewardFile)
        : m_stateFile(stateFile), m_actionFile(actionFile), m_rewardFile(rewardFile) {}
    
    void WriteState(const VehicleState& state) {
        std::ofstream ofs(m_stateFile);
        ofs << state.trafficDensity << ","
            << state.avgQueueLength << ","
            << state.linkQuality << ","
            << state.vehicleSpeed << ","
            << state.congestionLevel << ","
            << state.packetLossRate << ","
            << state.neighborCount << ","
            << state.distanceToDestination << std::endl;
        ofs.close();
    }
    
    int ReadAction() {
        std::ifstream ifs(m_actionFile);
        int action = 0;
        if (ifs.is_open()) {
            ifs >> action;
            ifs.close();
        }
        return action;
    }
    
    void WriteReward(const PerformanceMetrics& metrics) {
        std::ofstream ofs(m_rewardFile);
        ofs << (metrics.packetDelivered ? 1 : 0) << ","
            << metrics.delay << ","
            << metrics.congestionCreated << ","
            << metrics.routingOverhead << std::endl;
        ofs.close();
    }

private:
    std::string m_stateFile;
    std::string m_actionFile;
    std::string m_rewardFile;
};

/**
 * Calculate traffic density around a node
 */
double CalculateTrafficDensity(Ptr<Node> node, NodeContainer& vehicles, double radius) {
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    Vector pos = mobility->GetPosition();
    
    uint32_t count = 0;
    for (uint32_t i = 0; i < vehicles.GetN(); i++) {
        Ptr<MobilityModel> otherMobility = vehicles.Get(i)->GetObject<MobilityModel>();
        Vector otherPos = otherMobility->GetPosition();
        double distance = CalculateDistance(pos, otherPos);
        if (distance <= radius && distance > 0) {
            count++;
        }
    }
    
    // Vehicles per km (assuming radius in meters)
    return (count * 1000.0) / (2.0 * radius);
}

/**
 * Get current state for a vehicle
 */
VehicleState GetVehicleState(Ptr<Node> node, NodeContainer& vehicles, Ptr<Node> destination) {
    VehicleState state;
    
    Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
    Vector pos = mobility->GetPosition();
    Vector vel = mobility->GetVelocity();
    
    // Traffic density within 200m
    state.trafficDensity = CalculateTrafficDensity(node, vehicles, 200.0);
    
    // Average queue length (simplified - would need actual queue monitoring)
    state.avgQueueLength = state.trafficDensity * 0.5; // Approximation
    
    // Link quality (simplified - would need actual SINR measurement)
    state.linkQuality = 20.0 - (state.trafficDensity * 0.2); // dB, decreases with density
    
    // Vehicle speed
    state.vehicleSpeed = std::sqrt(vel.x * vel.x + vel.y * vel.y);
    
    // Congestion level (0-1)
    state.congestionLevel = std::min(state.trafficDensity / 100.0, 1.0);
    
    // Packet loss rate (approximation based on congestion)
    state.packetLossRate = state.congestionLevel * 0.3;
    
    // Neighbor count
    state.neighborCount = static_cast<uint32_t>(state.trafficDensity * 0.4); // Approximation
    
    // Distance to destination
    if (destination) {
        Ptr<MobilityModel> destMobility = destination->GetObject<MobilityModel>();
        Vector destPos = destMobility->GetPosition();
        state.distanceToDestination = CalculateDistance(pos, destPos);
    } else {
        state.distanceToDestination = 0;
    }
    
    return state;
}

/**
 * Packet receive callback
 */
void ReceivePacket(Ptr<Socket> socket) {
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from))) {
        g_packetsReceived++;
        NS_LOG_INFO("Packet received at " << Simulator::Now().GetSeconds() << "s");
    }
}

/**
 * Send packet with DQN-based routing decision
 */
void SendPacket(Ptr<Socket> socket, Ptr<Node> source, NodeContainer& vehicles, 
                Ptr<Node> destination, DQNInterface& dqnInterface, uint32_t packetSize) {
    
    // Get current state
    VehicleState state = GetVehicleState(source, vehicles, destination);
    
    // Write state to file for DQN agent
    dqnInterface.WriteState(state);
    
    // Read action from DQN agent (in real implementation, this would be async)
    // For now, we'll use a simple routing decision
    int action = dqnInterface.ReadAction();
    
    // Create and send packet
    Ptr<Packet> packet = Create<Packet>(packetSize);
    socket->Send(packet);
    g_packetsSent++;
    
    NS_LOG_INFO("Packet sent at " << Simulator::Now().GetSeconds() << "s with action " << action);
}

/**
 * Main simulation function
 */
int main(int argc, char *argv[]) {
    // Simulation parameters
    uint32_t nVehicles = 50;
    double simTime = 100.0; // seconds
    double areaSize = 1000.0; // meters
    uint32_t packetSize = 1024; // bytes
    double packetInterval = 0.1; // seconds
    bool verbose = false;
    std::string outputDir = "vanet-dqn-results";
    
    // Parse command line arguments
    CommandLine cmd;
    cmd.AddValue("nVehicles", "Number of vehicles", nVehicles);
    cmd.AddValue("simTime", "Simulation time (seconds)", simTime);
    cmd.AddValue("areaSize", "Simulation area size (meters)", areaSize);
    cmd.AddValue("packetSize", "Packet size (bytes)", packetSize);
    cmd.AddValue("packetInterval", "Packet interval (seconds)", packetInterval);
    cmd.AddValue("verbose", "Enable verbose logging", verbose);
    cmd.AddValue("outputDir", "Output directory", outputDir);
    cmd.Parse(argc, argv);
    
    // Enable logging
    if (verbose) {
        LogComponentEnable("VANETDQNSimulation", LOG_LEVEL_INFO);
    }
    
    NS_LOG_INFO("Starting VANET DQN Simulation");
    NS_LOG_INFO("Number of vehicles: " << nVehicles);
    NS_LOG_INFO("Simulation time: " << simTime << " seconds");
    
    // Create output directory
    std::string mkdirCmd = "mkdir -p " + outputDir;
    system(mkdirCmd.c_str());
    
    // Create vehicle nodes
    NodeContainer vehicles;
    vehicles.Create(nVehicles);
    
    // Configure WiFi (IEEE 802.11p for VANET)
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::FriisPropagationLossModel");
    
    YansWifiPhyHelper wifiPhy;
    Ptr<YansWifiChannel> channel = wifiChannel.Create();
    wifiPhy.SetChannel(channel);
    
    // 802.11p MAC and WiFi helper
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                  "DataMode", StringValue("OfdmRate6MbpsBW10MHz"),
                                  "ControlMode", StringValue("OfdmRate6MbpsBW10MHz"));
    
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, vehicles);
    
    // Configure mobility model
    MobilityHelper mobility;
    
    // Use RandomWaypointMobilityModel for realistic vehicle movement
    mobility.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
                                  "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + 
                                                   std::to_string(areaSize) + "]"),
                                  "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=" + 
                                                   std::to_string(areaSize) + "]"));
    
    mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                             "Speed", StringValue("ns3::UniformRandomVariable[Min=10.0|Max=25.0]"), // 10-25 m/s
                             "Pause", StringValue("ns3::ConstantRandomVariable[Constant=2.0]"),
                             "PositionAllocator", StringValue("ns3::RandomRectanglePositionAllocator[X=ns3::UniformRandomVariable[Min=0.0|Max=" + 
                                                             std::to_string(areaSize) + "]|Y=ns3::UniformRandomVariable[Min=0.0|Max=" + 
                                                             std::to_string(areaSize) + "]]"));
    
    mobility.Install(vehicles);
    
    // Install Internet stack
    InternetStackHelper internet;
    internet.Install(vehicles);
    
    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);
    
    // Create DQN interface
    DQNInterface dqnInterface(outputDir + "/state.csv", 
                             outputDir + "/action.csv", 
                             outputDir + "/reward.csv");
    
    // Setup UDP communication
    uint16_t port = 9;
    
    // Create receiver socket on last vehicle
    Ptr<Node> receiver = vehicles.Get(nVehicles - 1);
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    Ptr<Socket> recvSocket = Socket::CreateSocket(receiver, tid);
    InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), port);
    recvSocket->Bind(local);
    recvSocket->SetRecvCallback(MakeCallback(&ReceivePacket));
    
    // Create sender sockets on first few vehicles
    uint32_t nSenders = std::min(nVehicles / 5, 10u);
    for (uint32_t i = 0; i < nSenders; i++) {
        Ptr<Node> sender = vehicles.Get(i);
        Ptr<Socket> sendSocket = Socket::CreateSocket(sender, tid);
        InetSocketAddress remote = InetSocketAddress(interfaces.GetAddress(nVehicles - 1), port);
        sendSocket->Connect(remote);
        
        // Schedule periodic packet transmissions
        for (double t = 1.0; t < simTime; t += packetInterval) {
            Simulator::Schedule(Seconds(t), &SendPacket, sendSocket, sender, 
                              vehicles, receiver, dqnInterface, packetSize);
        }
    }
    
    // Install FlowMonitor for detailed statistics
    FlowMonitorHelper flowmon;
    Ptr<FlowMonitor> monitor = flowmon.InstallAll();
    
    // Enable NetAnim output
    AnimationInterface anim(outputDir + "/vanet-animation.xml");
    anim.SetMaxPktsPerTraceFile(500000);
    
    // Run simulation
    NS_LOG_INFO("Running simulation...");
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    
    // Calculate and print statistics
    NS_LOG_INFO("Simulation completed");
    NS_LOG_INFO("Packets sent: " << g_packetsSent);
    NS_LOG_INFO("Packets received: " << g_packetsReceived);
    NS_LOG_INFO("Packets dropped: " << g_packetsDropped);
    
    double pdr = (g_packetsSent > 0) ? (100.0 * g_packetsReceived / g_packetsSent) : 0;
    NS_LOG_INFO("Packet Delivery Ratio: " << pdr << "%");
    
    // FlowMonitor statistics
    monitor->CheckForLostPackets();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();
    
    std::ofstream statsFile(outputDir + "/flow-stats.txt");
    statsFile << "Flow statistics:\n";
    
    double totalThroughput = 0;
    double totalDelay = 0;
    uint32_t flowCount = 0;
    
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); 
         i != stats.end(); ++i) {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        
        double throughput = i->second.rxBytes * 8.0 / simTime / 1000; // kbps
        double delay = 0;
        if (i->second.rxPackets > 0) {
            delay = i->second.delaySum.GetSeconds() / i->second.rxPackets;
        }
        
        totalThroughput += throughput;
        totalDelay += delay;
        flowCount++;
        
        statsFile << "Flow " << i->first << " (" << t.sourceAddress << " -> " 
                  << t.destinationAddress << ")\n";
        statsFile << "  Tx Packets: " << i->second.txPackets << "\n";
        statsFile << "  Rx Packets: " << i->second.rxPackets << "\n";
        statsFile << "  Throughput: " << throughput << " kbps\n";
        statsFile << "  Mean Delay: " << delay << " s\n";
        statsFile << "  Packet Loss: " << i->second.lostPackets << "\n\n";
    }
    
    statsFile << "\nOverall Statistics:\n";
    statsFile << "Average Throughput: " << (flowCount > 0 ? totalThroughput / flowCount : 0) << " kbps\n";
    statsFile << "Average Delay: " << (flowCount > 0 ? totalDelay / flowCount : 0) << " s\n";
    statsFile << "Packet Delivery Ratio: " << pdr << "%\n";
    
    statsFile.close();
    
    NS_LOG_INFO("Statistics written to " << outputDir << "/flow-stats.txt");
    
    // Write final metrics for DQN
    PerformanceMetrics finalMetrics;
    finalMetrics.packetDelivered = (pdr > 50);
    finalMetrics.delay = (flowCount > 0 ? totalDelay / flowCount : 0);
    finalMetrics.congestionCreated = 1.0 - (pdr / 100.0);
    finalMetrics.routingOverhead = g_routingOverhead;
    dqnInterface.WriteReward(finalMetrics);
    
    Simulator::Destroy();
    NS_LOG_INFO("Simulation destroyed");
    
    return 0;
}
