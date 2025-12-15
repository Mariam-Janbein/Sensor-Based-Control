# Sensor-Based Control for Trajectory Tracking — Simulation and Hardware

## Overview
This repository contains software simulations and hardware experiments for trajectory tracking control of a quadrotor system. The control strategy implemented in this project is adopted from the following research article:

Robust Trajectory Tracking Control of a Quadrotor UAV Under Disturbances  
PLOS ONE  
Included in this repository as: journal.pone.0283195.pdf

Two reference trajectories from the paper were studied:
- Spiral
- Loop and corner

The work includes a full software evaluation of both trajectories and a hardware implementation using a DJI Tello drone for the spiral trajectory only.

<!-- ---

## Repository Structure

Sensor-Based-Control  
├── SoftwareSimulation/  
├── Results/  
│   ├── S1NoDisturbances/  
│   ├── S1-Disturbances/  
│   ├── S2NoDisturbances/  
│   └── S2-Disturbances/  
├── HardwareResults1/  
├── HardwareResult2/  
├── journal.pone.0283195.pdf  
└── README.md  

--- -->

## Software Simulation

### Description
Both trajectories proposed in the reference paper were implemented in Python and tested under two scenarios:
- Without disturbances
- With disturbances

The goal was to analyze robustness and tracking performance under external perturbations.

### Requirements
The following Python libraries must be installed:

pip install numpy scipy matplotlib

### How to Run the Simulation
1. Navigate to the simulation directory:
   cd SoftwareSimulation

2. Run the Python simulation scripts corresponding to the desired trajectory.

3. Simulation plots will be plotted on the screen.

NOTE: Change the return of the the function "def disturbances(t)" , to chose whether to run the code with or without disturbances.

---

## Simulation Results

The Results directory is organized as follows:

- S1NoDisturbances: Spiral trajectory without disturbances
- S1-Disturbances: Spiral trajectory with disturbances
- S2NoDisturbances: Loop and corner trajectory without disturbances
- S2-Disturbances: Loop and corner trajectory with disturbances

Key observations:
- Disturbances significantly degrade tracking performance.
- The Loop and corner trajectory is highly sensitive to disturbances.
- Even after tuning, Loop and corner results were not sufficient for hardware deployment.
- The Spiral trajectory demonstrated stable and promising simulation results.

---

## Hardware Implementation

### Platform
Hardware experiments were conducted using a DJI Tello drone.

Due to poor robustness observed in simulation, only the Spiral trajectory was implemented on hardware.

### Required Library
The drone is controlled using the DJITelloPy Python library:

https://github.com/damiafuentes/DJITelloPy

Installation:
pip install djitellopy

### How to Run the Hardware Code
1. Power on the DJI Tello drone.
2. Connect your computer to the drone via Wi-Fi.
3. Navigate to the hardware directory.
4. Run the main script:
   python Main.py

---

## Hardware Results

Two main behaviors were observed during real-world experiments:

1. The drone follows the spiral trajectory well, but flies too close to the ground.  
   This issue is mainly related to takeoff and altitude (z-axis) control.

2. The drone takes off successfully, reaches part of the trajectory, and then starts tracking it.  
   However, oscillations appear due to instability in the z-axis, resulting in an oscillating path.

The corresponding plots and experimental data are available in:
- HardwareResults1
- HardwareResult2

Further controller tuning is required to improve altitude stability and overall tracking performance.

---

## Conclusion
The Spiral trajectory showed promising results in both simulation and partial hardware implementation.  
The Loop and corner trajectory was found to be unsuitable for hardware testing due to sensitivity to disturbances.  
Future work should focus on improving z-axis stability and controller tuning for real-world deployment.

---

## Reference
Robust Trajectory Tracking Control of a Quadrotor UAV Under Disturbances  
PLOS ONE  
journal.pone.0283195.pdf
