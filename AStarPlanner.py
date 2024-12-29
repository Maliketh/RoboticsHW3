import numpy as np
import heapq


class AStarPlanner(object):
    def __init__(self, bb, start, goal):
        self.bb = bb
        self.start = start
        self.goal = goal

        self.nodes = dict()

        # used for visualizing the expanded nodes
        # make sure that this structure will contain a list of positions (states, numpy arrays) without duplicates
        self.expanded_nodes = []

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states (positions) of the robot.
        '''

        # initialize an empty plan.
        plan = []

        # define all directions the agent can take - order doesn't matter here
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (-1, 1), (1, 1), (1, -1)]

        self.epsilon = 20
        plan, cost = self.a_star(self.start, self.goal)
        return np.array(plan), cost

    # compute heuristic based on the planning_env
    def compute_heuristic(self, state):
        """
        Return the heuristic function for the A* algorithm.
        @param state The state (position) of the robot.
        """
        return self.epsilon * np.linalg.norm(np.array(state) - np.array(self.goal))

    def a_star(self, start_loc, goal_loc):
        """
        Perform A* search from start_loc to goal_loc.
        @param start_loc Starting location.
        @param goal_loc Goal location.
        Returns:
            path: List of nodes representing the path from start_loc to goal_loc.
            total_cost: The total cost of the path.
        """
        open_list = []
        closed_set = set()

        # Initialize g_cost (cost from start to node) and f_cost (g + heuristic)
        g_cost = {tuple(start_loc): 0}
        f_cost = {tuple(start_loc): self.compute_heuristic(start_loc)}

        # Push the start node to the priority queue
        heapq.heappush(open_list, (f_cost[tuple(start_loc)], tuple(start_loc)))

        # To reconstruct the path after reaching the goal
        came_from = {}

        while open_list:
            # Pop the node with the smallest f_cost
            _, current = heapq.heappop(open_list)

            # Add current node to expanded nodes for visualization
            self.expanded_nodes.append(np.array(current))

            # If the goal is reached, reconstruct and return the path and its cost
            if np.array_equal(current, goal_loc):
                path = self.reconstruct_path(came_from, current)
                total_cost = g_cost[tuple(goal_loc)]
                return path, total_cost

            closed_set.add(current)

            # Explore neighbors
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                cost = np.linalg.norm(np.array([dx, dy]))  # Cost of the action

                # Skip if neighbor is in the closed set or not valid
                if neighbor in closed_set or not self.bb.env.config_validity_checker(neighbor):
                    continue

                tentative_g_cost = g_cost[current] + cost

                if tuple(neighbor) not in g_cost or tentative_g_cost < g_cost[tuple(neighbor)]:
                    came_from[tuple(neighbor)] = current
                    g_cost[tuple(neighbor)] = tentative_g_cost
                    f_cost[tuple(neighbor)] = tentative_g_cost + self.compute_heuristic(neighbor)
                    heapq.heappush(open_list, (f_cost[tuple(neighbor)], tuple(neighbor)))

        # Return an empty list and infinite cost if no path is found
        return [], float('inf')

    def reconstruct_path(self, came_from, current):
        """
        Reconstruct the path from start to goal.
        @param came_from Dictionary tracking the path.
        @param current The current node (goal node).
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def get_expanded_nodes(self):
        '''
        Return list of expanded nodes without duplicates.
        '''

        # used for visualizing the expanded nodes
        return self.expanded_nodes
