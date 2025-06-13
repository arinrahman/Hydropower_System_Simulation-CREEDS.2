# 💧 HydroPower System Simulation

This repository simulates a scalable hydropower system, modeling the behavior of a reservoir and its energy generation components under varying environmental and operational conditions. The simulation is designed for extensibility and educational use, with a focus on real-world hydropower concepts.

##   Key Features

- **Reservoir and Turbine Modeling**  
  Simulates core components such as water storage, turbine discharge, and energy conversion.

- **Energy Balancing**  
  Calculates optimized energy release patterns based on inflow, demand, and solar input.

- **Web-Based Interface**  
  Lightweight Flask-based app for visualizing metrics and interacting with the simulation.

- **Data Integration**  
  Uses real discharge and storage datasets (e.g., IBWC) for input and experimentation.

##   Project Components

- `Simulator.py`: Central controller that runs the simulation.  
- `Reservoir.py`, `Turbine.py`, `Solar.py`: Individual modules representing physical subsystems.  
- `Energy_Balancer.py`: Coordinates energy flow and resource usage.  
- `Systems/`: Utility modules for solar generation, hydro release logic, day-specific configurations, and more.  
- `templates/` & `static/`: Web UI files for interacting with the system.

## ℹ️ Notes

The machine learning model files were intentionally removed to streamline the experience for new users. This allows users to focus first on the simulation and data analysis before building their own predictive models. The cleaned project structure ensures that users can get started right away after creating a new branch.

## 👥 Contributors

- **Arin Rahman** 
- **Jose Vega** 
