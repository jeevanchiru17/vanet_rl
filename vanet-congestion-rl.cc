/*
 * VANET Traffic Congestion Simulation with RL Support
 * Based on Q-learning approach for congestion control
 * Generates pcap traces for RL training
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/olsr-helper.h"
#include "ns3/dsdv-helper.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/ipv4-list-routing-helper.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/vanet-routing-compare-helper.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("VanetCongestionRL");

// Statistics collection
struct VehicleStats {
    uint32_t nodeId;
    double x;
    double y;
    double speed;
    uint32_t packetsSent;
    uint32_t packetsReceived;
    double cbr;  // Channel Busy Ratio
    double beaconRate;
    double txPower;
    Time lastUpdate;
};

std::map<uint32_t, VehicleStats> vehicleStats;
uint32_t totalPacketsSent = 0;
uint32_t totalPacketsReceived = 0;
uint32_t totalCollisions = 0;

// Callback for packet transmission
void TxCallback(std::string context, Ptr<const Packet> packet) {
    totalPacketsSent++;
    uint32_t nodeId = std::stoi(context.substr(context.find_last_of("/") + 1));
    if (vehicleStats.find(nodeId) != vehicleStats.end()) {
        vehicleStats[nodeId].packetsSent++;
    }
}

// Callback for packet reception
void RxCallback(std::string context, Ptr<const Packet> packet, const Address& address) {
    totalPacketsReceived++;
    uint32_t nodeId = std::stoi(context.substr(context.find_last_of("/") + 1));
    if (vehicleStats.find(nodeId) != vehicleStats.end()) {
        vehicleStats[nodeId].packetsReceived++;
    }
}

// Callback for packet drops (collisions)
void DropCallback(std::string context, Ptr<const Packet> packet, DropReason reason) {
    totalCollisions++;
}

// Calculate CBR (Channel Busy Ratio) for a node
void CalculateCBR(Ptr<Node> node, double& cbr) {
    Ptr<NetDevice> device = node->GetDevice(0);
    Ptr<WifiNetDevice> wifiDevice = DynamicCast<WifiNetDevice>(device);
    if (wifiDevice) {
        Ptr<WifiPhy> phy = wifiDevice->GetPhy();
        // Simplified CBR calculation - in real implementation, use actual channel sensing
        // This is a placeholder - actual CBR should be calculated from channel state
        cbr = 0.0; // Will be updated based on actual channel measurements
    }
}

// Update vehicle statistics periodically
void UpdateStatistics(NodeContainer& nodes, double simulationTime) {
    for (uint32_t i = 0; i < nodes.GetN(); i++) {
        Ptr<Node> node = nodes.Get(i);
        Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
        
        if (mobility) {
            Vector position = mobility->GetPosition();
            Vector velocity = mobility->GetVelocity();
            double speed = std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
            
            if (vehicleStats.find(i) == vehicleStats.end()) {
                VehicleStats stats;
                stats.nodeId = i;
                stats.packetsSent = 0;
                stats.packetsReceived = 0;
                stats.cbr = 0.0;
                stats.beaconRate = 10.0; // Default 10 Hz
                stats.txPower = 20.0; // Default 20 dBm
                vehicleStats[i] = stats;
            }
            
            vehicleStats[i].x = position.x;
            vehicleStats[i].y = position.y;
            vehicleStats[i].speed = speed;
            vehicleStats[i].lastUpdate = Simulator::Now();
            
            // Calculate CBR
            CalculateCBR(node, vehicleStats[i].cbr);
        }
    }
    
    // Schedule next update
    Simulator::Schedule(Seconds(0.1), &UpdateStatistics, nodes, simulationTime);
}

// Print statistics to file
void PrintStatistics(double simulationTime) {
    std::ofstream statsFile;
    statsFile.open("vanet_statistics.txt", std::ios::out | std::ios::app);
    
    statsFile << "Time: " << Simulator::Now().GetSeconds() << "s\n";
    statsFile << "Total Packets Sent: " << totalPacketsSent << "\n";
    statsFile << "Total Packets Received: " << totalPacketsReceived << "\n";
    statsFile << "Total Collisions: " << totalCollisions << "\n";
    statsFile << "Packet Delivery Ratio: " 
              << (totalPacketsSent > 0 ? (double)totalPacketsReceived / totalPacketsSent : 0.0) 
              << "\n\n";
    
    statsFile.close();
}

int main (int argc, char *argv[])
{
    // Simulation parameters
    uint32_t nVehicles = 50;
    double simulationTime = 100.0;
    double minSpeed = 10.0;  // m/s
    double maxSpeed = 30.0;  // m/s
    double beaconInterval = 0.1; // 10 Hz default
    double txPower = 20.0;   // dBm
    std::string phyMode = "OfdmRate6MbpsBW10MHz";
    bool enablePcap = true;
    std::string pcapPrefix = "vanet-congestion";
    bool enableFlowMonitor = true;
    std::string traceFile = "vanet-flowmon.xml";
    
    // Command line arguments
    CommandLine cmd;
    cmd.AddValue("nVehicles", "Number of vehicles", nVehicles);
    cmd.AddValue("simTime", "Simulation time in seconds", simulationTime);
    cmd.AddValue("minSpeed", "Minimum vehicle speed (m/s)", minSpeed);
    cmd.AddValue("maxSpeed", "Maximum vehicle speed (m/s)", maxSpeed);
    cmd.AddValue("beaconInterval", "Beacon interval in seconds", beaconInterval);
    cmd.AddValue("txPower", "Transmission power (dBm)", txPower);
    cmd.AddValue("phyMode", "Physical layer mode", phyMode);
    cmd.AddValue("enablePcap", "Enable pcap tracing", enablePcap);
    cmd.AddValue("pcapPrefix", "Pcap file prefix", pcapPrefix);
    cmd.Parse(argc, argv);
    
    // Enable logging
    LogComponentEnable("VanetCongestionRL", LOG_LEVEL_INFO);
    // LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
    // LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
    
    // Create nodes (vehicles)
    NodeContainer vehicles;
    vehicles.Create(nVehicles);
    
    // Create mobility model (Random Waypoint)
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
                                   "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=1000.0]"),
                                   "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=1000.0]"));
    
    mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                              "Speed", StringValue("ns3::UniformRandomVariable[Min=" + 
                                                   std::to_string(minSpeed) + "|Max=" + 
                                                   std::to_string(maxSpeed) + "]"),
                              "Pause", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"),
                              "PositionAllocator", StringValue("ns3::RandomRectanglePositionAllocator"));
    
    mobility.Install(vehicles);
    
    // Configure WiFi (802.11p)
    std::string phyModeString = phyMode;
    Config::SetDefault("ns3::WifiRemoteStationManager::NonUnicastMode", StringValue(phyModeString));
    
    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::FriisPropagationLossModel");
    
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(wifiChannel.Create());
    wifiPhy.Set("TxPowerStart", DoubleValue(txPower));
    wifiPhy.Set("TxPowerEnd", DoubleValue(txPower));
    wifiPhy.Set("TxPowerLevels", UintegerValue(1));
    wifiPhy.Set("RxGain", DoubleValue(0));
    wifiPhy.Set("RxNoiseFigure", DoubleValue(10));
    wifiPhy.SetErrorRateModel("ns3::NistErrorRateModel");
    
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211p);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                  "DataMode", StringValue(phyModeString),
                                  "ControlMode", StringValue(phyModeString));
    
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    
    NetDeviceContainer devices = wifi.Install(wifiPhy, wifiMac, vehicles);
    
    // Install Internet stack
    InternetStackHelper internet;
    internet.Install(vehicles);
    
    // Assign IP addresses
    Ipv4AddressHelper ipv4;
    ipv4.SetBase("10.1.1.0", "255.255.255.0");
    Ipv4InterfaceContainer interfaces = ipv4.Assign(devices);
    
    // Install OLSR routing
    OlsrHelper olsr;
    Ipv4StaticRoutingHelper staticRouting;
    Ipv4ListRoutingHelper list;
    list.Add(staticRouting, 0);
    list.Add(olsr, 10);
    internet.SetRoutingHelper(list);
    
    // Create UDP applications for beaconing (BSM messages)
    uint16_t port = 9;
    ApplicationContainer serverApps;
    ApplicationContainer clientApps;
    
    for (uint32_t i = 0; i < vehicles.GetN(); i++) {
        // Server (receiver) on each vehicle
        UdpEchoServerHelper server(port);
        serverApps.Add(server.Install(vehicles.Get(i)));
        
        // Client (sender) on each vehicle - broadcasts beacons
        UdpEchoClientHelper client(Ipv4Address("255.255.255.255"), port);
        client.SetAttribute("MaxPackets", UintegerValue(1000000));
        client.SetAttribute("Interval", TimeValue(Seconds(beaconInterval)));
        client.SetAttribute("PacketSize", UintegerValue(200)); // BSM size ~200 bytes
        clientApps.Add(client.Install(vehicles.Get(i)));
    }
    
    serverApps.Start(Seconds(1.0));
    serverApps.Stop(Seconds(simulationTime));
    clientApps.Start(Seconds(2.0));
    clientApps.Stop(Seconds(simulationTime));
    
    // Enable pcap tracing
    if (enablePcap) {
        wifiPhy.EnablePcap(pcapPrefix, devices);
    }
    
    // Setup callbacks for statistics
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/TxBegin", 
                    MakeCallback(&TxCallback));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/RxEnd", 
                    MakeCallback(&RxCallback));
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyRxDrop", 
                    MakeCallback(&DropCallback));
    
    // Flow monitor
    Ptr<FlowMonitor> flowMonitor;
    FlowMonitorHelper flowHelper;
    if (enableFlowMonitor) {
        flowMonitor = flowHelper.InstallAll();
    }
    
    // Schedule statistics updates
    Simulator::Schedule(Seconds(1.0), &UpdateStatistics, vehicles, simulationTime);
    Simulator::Schedule(Seconds(10.0), &PrintStatistics, simulationTime);
    
    // Run simulation
    std::cout << "Starting simulation with " << nVehicles << " vehicles for " 
              << simulationTime << " seconds...\n";
    Simulator::Stop(Seconds(simulationTime));
    Simulator::Run();
    
    // Print final statistics
    std::cout << "\n=== Simulation Results ===\n";
    std::cout << "Total Packets Sent: " << totalPacketsSent << "\n";
    std::cout << "Total Packets Received: " << totalPacketsReceived << "\n";
    std::cout << "Total Collisions: " << totalCollisions << "\n";
    if (totalPacketsSent > 0) {
        std::cout << "Packet Delivery Ratio: " 
                  << (double)totalPacketsReceived / totalPacketsSent << "\n";
    }
    
    // Flow monitor statistics
    if (enableFlowMonitor) {
        flowMonitor->SerializeToXmlFile(traceFile, true, true);
        std::cout << "Flow monitor trace saved to: " << traceFile << "\n";
    }
    
    Simulator::Destroy();
    
    std::cout << "Simulation completed. Pcap files generated with prefix: " 
              << pcapPrefix << "\n";
    std::cout << "Use pcap_to_csv.py to convert pcap files to CSV for RL training.\n";
    
    return 0;
}

