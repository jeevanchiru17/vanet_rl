/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Simple VANET NetSimulyzer Test
 *
 * This example demonstrates basic NetSimulyzer integration with a VANET scenario.
 * It creates a few vehicles, shows their movement, and logs basic information.
 */

#include "ns3/core-module.h"
#include "ns3/mobility-module.h"
#include "ns3/netsimulyzer-module.h"
#include "ns3/network-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("VanetNetSimulyzerTest");

int
main(int argc, char* argv[])
{
    // Enable logging
    LogComponentEnable("VanetNetSimulyzerTest", LOG_LEVEL_INFO);

    // Simulation parameters
    uint32_t numVehicles = 10;
    double simTime = 30.0; // seconds
    std::string outputFile = "vanet-test.json";

    // Parse command line arguments
    CommandLine cmd;
    cmd.AddValue("vehicles", "Number of vehicles", numVehicles);
    cmd.AddValue("simTime", "Simulation time in seconds", simTime);
    cmd.AddValue("output", "Output JSON file name", outputFile);
    cmd.Parse(argc, argv);

    NS_LOG_INFO("Creating VANET NetSimulyzer test with " << numVehicles << " vehicles");

    // Create NetSimulyzer orchestrator
    auto orchestrator = CreateObject<netsimulyzer::Orchestrator>(outputFile);

    // Create vehicle nodes
    NodeContainer vehicles;
    vehicles.Create(numVehicles);

    // Set up mobility model for vehicles (random direction on a road)
    auto positionAllocator = CreateObject<RandomRectanglePositionAllocator>();
    auto xVar = CreateObject<UniformRandomVariable>();
    xVar->SetAttribute("Min", DoubleValue(0.0));
    xVar->SetAttribute("Max", DoubleValue(1000.0));
    auto yVar = CreateObject<UniformRandomVariable>();
    yVar->SetAttribute("Min", DoubleValue(0.0));
    yVar->SetAttribute("Max", DoubleValue(100.0));

    positionAllocator->SetX(xVar);
    positionAllocator->SetY(yVar);

    auto speedVar = CreateObject<UniformRandomVariable>();
    speedVar->SetAttribute("Min", DoubleValue(10.0));
    speedVar->SetAttribute("Max", DoubleValue(30.0));

    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::RandomDirection2dMobilityModel",
                              "Bounds",
                              RectangleValue(Rectangle(0.0, 1000.0, 0.0, 100.0)),
                              "Speed",
                              PointerValue(speedVar),
                              "Pause",
                              StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));
    mobility.SetPositionAllocator(positionAllocator);
    mobility.Install(vehicles);

    // Configure NetSimulyzer visualization for vehicles
    netsimulyzer::NodeConfigurationHelper nodeHelper{orchestrator};
    nodeHelper.Set("Model", StringValue(netsimulyzer::models::CAR));
    nodeHelper.Set("EnableMotionTrail", BooleanValue(true));
    nodeHelper.Install(vehicles);

    // Create a log stream for simulation events
    auto infoLog = CreateObject<netsimulyzer::LogStream>(orchestrator);
    infoLog->SetAttribute("Name", StringValue("Simulation Info"));

    *infoLog << "VANET Simulation Started\\n";
    *infoLog << "Number of vehicles: " << numVehicles << "\\n";
    *infoLog << "Simulation time: " << simTime << " seconds\\n";

    // Create a series to track number of active vehicles over time
    auto vehicleCountSeries = CreateObject<netsimulyzer::XYSeries>(orchestrator);
    vehicleCountSeries->SetAttribute("Name", StringValue("Active Vehicles"));
    vehicleCountSeries->SetAttribute("Connection", EnumValue(netsimulyzer::XYSeries::Line));

    // Log vehicle count at regular intervals
    for (double t = 0; t <= simTime; t += 1.0)
    {
        Simulator::Schedule(Seconds(t), [vehicleCountSeries, numVehicles, t]() {
            vehicleCountSeries->Append(t, numVehicles);
        });
    }

    // Define a road area
    Rectangle roadArea{0.0, 1000.0, 0.0, 100.0};
    auto road = CreateObject<netsimulyzer::RectangularArea>(orchestrator, roadArea);
    road->SetAttribute("BorderColor", netsimulyzer::Color3Value{128u, 128u, 128u});
    road->SetAttribute("FillColor", netsimulyzer::Color3Value{200u, 200u, 200u});

    // Log periodic position updates
    for (uint32_t i = 0; i < numVehicles; i++)
    {
        Simulator::Schedule(Seconds(5.0), [infoLog, &vehicles, i]() {
            Ptr<MobilityModel> mob = vehicles.Get(i)->GetObject<MobilityModel>();
            Vector pos = mob->GetPosition();
            *infoLog << "Vehicle " << i << " at position (" << pos.x << ", " << pos.y << ")\\n";
        });
    }

    // Schedule simulation end log
    Simulator::Schedule(Seconds(simTime), [infoLog, simTime]() {
        *infoLog << "Simulation completed at " << simTime << " seconds\\n";
    });

    NS_LOG_INFO("Starting simulation...");

    // Run simulation
    Simulator::Stop(Seconds(simTime));
    Simulator::Run();
    Simulator::Destroy();

    NS_LOG_INFO("Simulation complete. Output written to: " << outputFile);
    NS_LOG_INFO("Open this file in the NetSimulyzer application to view the 3D visualization.");

    return 0;
}
