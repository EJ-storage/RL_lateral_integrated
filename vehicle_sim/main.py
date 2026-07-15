"""
Main entry point for vehicle dynamics simulation
Example usage and demonstration
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from simulator import VehicleSimulator, SimulatorConfig
from controllers.driver_controller import DriverController
from scenarios.straight_line_scenario import StraightLineScenario
from scenarios.constant_radius_scenario import ConstantRadiusScenario


def run_straight_line_test():
    """Run straight line acceleration and braking test"""
    print("Running straight line acceleration/braking test...")

    # Create simulator
    sim_config = SimulatorConfig(
        dt=0.001,
        max_time=15.0,
        enable_logging=True,
        log_rate=10
    )
    simulator = VehicleSimulator(sim_config)

    # Create scenario
    scenario_config = {
        'accel_duration': 5.0,
        'coast_duration': 2.0,
        'brake_duration': 3.0,
        'throttle_level': 0.8,
        'brake_level': 0.6
    }
    scenario = StraightLineScenario(scenario_config)

    # Create controller
    controller_config = {
        'max_motor_torque': 250.0,
        'max_brake_torque': 2000.0,
        'wheelbase': 2.7,
        'track_width': 1.6
    }
    controller = DriverController(controller_config)

    # Setup simulation
    simulator.set_scenario(scenario)
    simulator.set_controller(controller)

    # Run simulation
    results = simulator.run()

    print(f"Simulation complete. Total time: {simulator.time:.2f}s")

    # TODO: Plot results
    # plot_straight_line_results(results)

    return results


def run_cornering_test():
    """Run constant radius cornering test"""
    print("Running constant radius cornering test...")

    # Create simulator
    sim_config = SimulatorConfig(
        dt=0.001,
        max_time=15.0,
        enable_logging=True,
        log_rate=10
    )
    simulator = VehicleSimulator(sim_config)

    # Create scenario
    scenario_config = {
        'entry_duration': 3.0,
        'cornering_duration': 5.0,
        'exit_duration': 2.0,
        'target_speed': 15.0,
        'corner_radius': 50.0,
        'direction': 'left',
        'wheelbase': 2.7
    }
    scenario = ConstantRadiusScenario(scenario_config)

    # Create controller
    controller_config = {
        'max_motor_torque': 250.0,
        'wheelbase': 2.7,
        'track_width': 1.6
    }
    controller = DriverController(controller_config)

    # Setup simulation
    simulator.set_scenario(scenario)
    simulator.set_controller(controller)

    # Run simulation
    results = simulator.run()

    print(f"Simulation complete. Total time: {simulator.time:.2f}s")

    # TODO: Plot results
    # plot_cornering_results(results)

    return results


def plot_straight_line_results(results: Any):
    """Plot straight line test results"""
    # TODO: Implement plotting
    # Plot velocity, acceleration, distance vs time
    pass


def plot_cornering_results(results: Any):
    """Plot cornering test results"""
    # TODO: Implement plotting
    # Plot trajectory, lateral acceleration, yaw rate vs time
    pass


def main():
    """Main function"""
    print("=" * 60)
    print("Vehicle Dynamics Simulation")
    print("=" * 60)
    print()

    # Run different test scenarios
    try:
        # Straight line test
        print("Test 1: Straight Line Acceleration/Braking")
        print("-" * 60)
        straight_results = run_straight_line_test()
        print()

        # Cornering test
        print("Test 2: Constant Radius Cornering")
        print("-" * 60)
        cornering_results = run_cornering_test()
        print()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
