import numpy as np
import matplotlib.pyplot as plt

import os
from datetime import datetime
import time
import json

from twoD.environment import MapEnvironment
from twoD.dot_environment import MapDotEnvironment
from twoD.dot_building_blocks import DotBuildingBlocks2D
from twoD.building_blocks import BuildingBlocks2D
from twoD.dot_visualizer import DotVisualizer
from threeD.environment import Environment
from threeD.kinematics import UR5e_PARAMS, Transform
from threeD.building_blocks import BuildingBlocks3D
from threeD.visualizer import Visualize_UR
from AStarPlanner import AStarPlanner
from RRTMotionPlanner import RRTMotionPlanner
from RRTInspectionPlanner import RRTInspectionPlanner
from RRTStarPlanner import RRTStarPlanner
from twoD.visualizer import Visualizer

# MAP_DETAILS = {"json_file": "twoD/map1.json", "start": np.array([10,10]), "goal": np.array([4, 6])}
MAP_DETAILS = {"json_file": "twoD/map2.json", "start": np.array([360, 150]), "goal": np.array([100, 200])}

def run_dot_2d_astar():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = AStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, expanded_nodes=planner.get_expanded_nodes(), show_map=True, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

def run_dot_2d_rrt():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"],
                               ext_mode="E1", goal_prob=0.01, step_size=0.5)

    # execute plan
    plan = planner.plan()
    DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_2d_rrt_motion_planning():
    MAP_DETAILS = {"json_file": "twoD/map_mp.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="mp")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E2", goal_prob=0.01, step_size=0.5)
    # execute plan
    plan = planner.plan()
    #Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"])

    return plan

def run_2d_rrt_inspection_planning():
    MAP_DETAILS = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0]), "goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)
    planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E2", goal_prob=0.01, coverage=0.5)

    # execute plan
    plan = np.array(planner.plan())
    Visualizer(bb).visualize_plan(plan=plan, start=MAP_DETAILS["start"])

def run_dot_2d_rrt_star():
    planning_env = MapDotEnvironment(json_file=MAP_DETAILS["json_file"])
    bb = DotBuildingBlocks2D(planning_env)
    planner = RRTStarPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E1", goal_prob=0.01, k=5, max_step_size=None, max_itr=1000)

    # execute plan
    plan, improvements = planner.plan2()
    if plan is not None:
        # Unpack costs and iterations
        costs, iterations = zip(*improvements)

        # Plot cost vs. iteration graph
        plt.figure(figsize=(8, 6))
        plt.plot(iterations, costs, marker='o', linestyle='-', color='b', label="Cost Improvement")
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost vs Iteration')
        plt.grid(True)
        plt.legend()
        plt.show()
        # visualizer.show_path(path)
    #DotVisualizer(bb).visualize_map(plan=plan, tree_edges=planner.tree.get_edges_as_states(), show_map=True)

def run_3d():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.1)

    #visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0])
    # ---------------------------------------
    path = None
    while path is None:
        rrt_star_planner = RRTStarPlanner(max_step_size=0.4,
                                          start=env2_start,
                                          goal=env2_goal,
                                          max_itr=1600,
                                          stop_on_goal=False,
                                          bb=bb,
                                          goal_prob=0.05,
                                          ext_mode="E2",
                                          k=5)

        path, costs_improvement = rrt_star_planner.plan2()
        if path is not None:
            # Unpack costs and iterations
            costs, iterations = zip(*costs_improvement)

            # Plot cost vs. iteration graph
            plt.figure(figsize=(8, 6))
            plt.plot(iterations, costs, marker='o', linestyle='-', color='b', label="Cost Improvement")
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Cost vs Iteration')
            plt.grid(True)
            plt.legend()
            plt.show()
            # visualizer.show_path(path)

def benchmark2D(p, goal_probs, num_runs):
    MAP_DETAILS = {"json_file": "twoD/map_ip.json", "start": np.array([0.78, -0.78, 0.0, 0.0]),"goal": np.array([0.3, 0.15, 1.0, 1.1])}
    planning_env = MapEnvironment(json_file=MAP_DETAILS["json_file"], task="ip")
    bb = BuildingBlocks2D(planning_env)

    for prob in goal_probs:
        times = []
        costs = []
        for i in range(num_runs):
            print(".", end='')
            if p == "IP":
                planner = RRTInspectionPlanner(bb=bb, start=MAP_DETAILS["start"], ext_mode="E1", goal_prob=0.05, coverage=prob, step_size=0.5)
            else:
                planner = RRTMotionPlanner(bb=bb, start=MAP_DETAILS["start"], goal=MAP_DETAILS["goal"], ext_mode="E1",
                                           goal_prob=prob, step_size=0.5)
            tic = time.time()
            plan = np.array(planner.plan())
            tac = time.time()

            times.append(tac - tic)
            costs.append(bb.compute_path_cost(plan))
        print(f"\nAverage time: {np.mean(times)}"
              f"\nAverage cost: {np.mean(costs)}")

def benchmark3D():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    bb = BuildingBlocks3D(transform=transform,
                          ur_params=ur_params,
                          env=env,
                          resolution=0.1)

    #visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)

    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0])
    # ---------------------------------------
    num_tests = 16
    for max_step_size in [0.05, 0.1, 0.2, 0.25, 0.4]:
        for p in [0.05, 0.2]:
            print(f"\n-----------------Step: {max_step_size}, prob: {p}-----------------")
            times = []
            costs = []
            plans = []
            improvements = []

            for i in range(num_tests):
                print(".", end='')

                rrt_star_planner = RRTStarPlanner(max_step_size=max_step_size,
                                                  start=env2_start,
                                                  goal=env2_goal,
                                                  max_itr=1600,
                                                  stop_on_goal=False,
                                                  bb=bb,
                                                  goal_prob=p,
                                                  ext_mode="E2",
                                                  k=5)

                tic = time.time()
                path, cost_improvement = rrt_star_planner.plan2()
                tac = time.time()
                if path is not None:
                    times.append(tac - tic)
                    costs.append(rrt_star_planner.compute_cost(path))
                    plans.append(path)
                    improvements.append(cost_improvement)

            average_time = sum(times) / len(times) if len(times) > 0 else 0
            average_cost = sum(costs) / len(costs) if len(costs) > 0 else 0
            best_plan_index = costs.index(min(costs)) if len(costs) > 0 else -1
            best_plan = plans[best_plan_index] if best_plan_index != -1 else None

            res = {
                "parameters": {
                    "step_size": max_step_size,
                    "goal_prob": p
                },
                "average_time": average_time,
                "average_cost": average_cost,
                "best_plan": best_plan.tolist() if (best_plan is not None) else None,
                "success_rate": 100 * len(times) / num_tests,
                "improvements": improvements
            }

            # Create a filename based on the parameters
            filename = f"step_{max_step_size}_goal_{p}.json"
            filepath = os.path.join("./exps", filename)

            # Save the dictionary as a JSON file
            with open(filepath, 'w') as f:
                json.dump(res, f, indent=4)

def get_benchmark_results(directory):
    def load_data(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    goal_probs = [0.05, 0.2]
    data_by_goal = {goal_prob: {"times": [], "costs": [], "success_rates": [], "step_sizes": []} for goal_prob in
                    goal_probs}

    # Iterate over the JSON files to load the data
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            data = load_data(file_path)

            # Extract parameters from the data
            goal_prob = data["parameters"]["goal_prob"]
            max_step_size = data["parameters"]["step_size"]
            average_time = data["average_time"]
            average_cost = data["average_cost"]
            success_rate = data["success_rate"]

            # Store the data
            if goal_prob in data_by_goal:
                data_by_goal[goal_prob]["times"].append(average_time)
                data_by_goal[goal_prob]["costs"].append(average_cost)
                data_by_goal[goal_prob]["success_rates"].append(success_rate)
                data_by_goal[goal_prob]["step_sizes"].append(max_step_size)

    return data_by_goal

def plot_benchmark_results(goal_prob, data_by_goal):
    # Extract the data for the given goal_prob
    times = data_by_goal[goal_prob]["times"]
    costs = data_by_goal[goal_prob]["costs"]
    success_rates = data_by_goal[goal_prob]["success_rates"]
    step_sizes = data_by_goal[goal_prob]["step_sizes"]

    # Sort the data by computation time (x-axis)
    sorted_data = sorted(zip(times, costs, success_rates, step_sizes), key=lambda x: x[0])
    times, costs, success_rates, step_sizes = zip(*sorted_data)

    # Plot for Cost vs. Computation Time for this goal_prob
    plt.figure(figsize=(10, 5))
    plt.plot(times, costs, marker='o', label=f'Goal Prob {goal_prob}')
    for i, step_size in enumerate(step_sizes):
        plt.text(times[i], costs[i], f'{step_size:.2f}', fontsize=9, ha='right')
    plt.xlabel('Computation Time (seconds)')
    plt.ylabel('Average Cost')
    plt.title(f'Cost vs. Computation Time (Goal Prob {goal_prob})')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot for Success Rate vs. Computation Time for this goal_prob
    plt.figure(figsize=(10, 5))
    plt.plot(times, success_rates, marker='o', label=f'Goal Prob {goal_prob}')
    for i, step_size in enumerate(step_sizes):
        plt.text(times[i], success_rates[i], f'{step_size:.3f}', fontsize=9, ha='right')
    plt.xlabel('Computation Time (seconds)')
    plt.ylabel('Success Rate (%)')
    plt.title(f'Success Rate vs. Computation Time (Goal Prob {goal_prob})')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #benchmark2D("IP", [0.5, 0.75], 10)
    #benchmark2D("MP", [0.05, 0.2], 1)
    benchmark3D()

    '''
    path = "./exps"
    data_by_goal = get_benchmark_results(path)
    plans = []
    for p in [0.05, 0.2]:
        plot_benchmark_results(p, data_by_goal)
    '''

    #run_dot_2d_astar()
    #run_dot_2d_rrt()
    #run_dot_2d_rrt_star()
    #run_2d_rrt_inspection_planning()
    #run_3d()
