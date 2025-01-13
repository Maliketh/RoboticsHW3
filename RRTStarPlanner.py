import numpy as np
from RRTTree import RRTTree
import time


class RRTStarPlanner(object):

    def __init__(self, bb, ext_mode, max_step_size, start, goal,
                 max_itr=None, stop_on_goal=None, k=None, goal_prob=0.01):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        self.max_itr = max_itr
        self.stop_on_goal = stop_on_goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.k = k

        self.step_size = max_step_size

    def get_safe_neighbors(self, new_config):
        """
        Safely get k nearest neighbors with bounds checking
        """
        num_vertices = len(self.tree.vertices)
        if num_vertices <= 1:  # If tree only has start node
            return [], []

        # Never request more neighbors than available vertices minus current
        k = min(self.k, num_vertices - 1)
        if k < 1:  # Safety check
            return [], []

        try:
            near_ids, near_configs = self.tree.get_k_nearest_neighbors(new_config, k)
            # Additional validation of returned lists
            if len(near_ids) != len(near_configs) or len(near_ids) == 0:
                return [], []
            return near_ids, near_configs
        except Exception as e:
            print(f"Error getting neighbors: {e}")
            return [], []

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        # Tree sampling
        self.tree.add_vertex(self.start)

        while not self.tree.is_goal_exists(self.goal):
            rand_config = self.sample_random_config(self.goal_prob, self.goal)
            near_config_id, near_config = self.tree.get_nearest_config(rand_config)
            new_config = self.extend(near_config, rand_config)

            if new_config is not None and self.bb.edge_validity_checker(new_config, near_config):
                # Get k nearest neighbors safely
                near_ids, near_configs = self.get_safe_neighbors(new_config)

                # Only proceed with RRT* optimizations if we have neighbors
                if near_ids:
                    # Find best parent among nearest neighbors
                    min_cost = float('inf')
                    best_parent_id = None

                    for idx, near_id in enumerate(near_ids):
                        if self.bb.edge_validity_checker(new_config, near_configs[idx]):
                            cost = self.bb.compute_distance(new_config, near_configs[idx])
                            total_cost = self.tree.vertices[near_id].cost + cost
                            if total_cost < min_cost:
                                min_cost = total_cost
                                best_parent_id = near_id

                    if best_parent_id is not None:
                        # Add vertex and edge with best parent
                        new_vid = self.tree.add_vertex(new_config)
                        cost = self.bb.compute_distance(new_config, self.tree.vertices[best_parent_id].config)
                        self.tree.add_edge(best_parent_id, new_vid, cost)

                        # Rewire nearby nodes if new node provides better path
                        for idx, near_id in enumerate(near_ids):
                            if near_id != new_vid and self.bb.edge_validity_checker(new_config, near_configs[idx]):
                                cost = self.bb.compute_distance(near_configs[idx], new_config)
                                potential_cost = self.tree.vertices[new_vid].cost + cost
                                if potential_cost < self.tree.vertices[near_id].cost:
                                    self.tree.edges[near_id] = new_vid
                                    self.tree.vertices[near_id].set_cost(potential_cost)
                                    self.update_children_costs(near_id)
                else:
                    # If no valid neighbors found, fall back to basic RRT behavior
                    new_vid = self.tree.add_vertex(new_config)
                    cost = self.bb.compute_distance(new_config, near_config)
                    self.tree.add_edge(near_config_id, new_vid, cost)

        print(f"Tree is generated and contains: {len(self.tree.vertices)} vertices")

        # Plan computation
        goal_idx = self.tree.get_idx_for_config(self.goal)
        if goal_idx is None:
            print("Goal configuration not found in tree")
            return None

        plan = [self.goal]
        current_idx = goal_idx

        try:
            while not np.allclose(self.tree.vertices[current_idx].config, self.start, atol=1e-6):
                parent_idx = self.tree.edges[current_idx]
                parent_config = self.tree.vertices[parent_idx].config
                plan.insert(0, parent_config)
                current_idx = parent_idx

            # Add start configuration if not already included
            if not np.allclose(plan[0], self.start, atol=1e-6):
                plan.insert(0, self.start)

            return np.array(plan)
        except KeyError as e:
            print(f"No path found between {self.start} and {self.goal}")
            return None

    def update_children_costs(self, parent_id):
        '''
        Recursively update the costs of children after rewiring
        '''
        children = [vid for vid, pid in self.tree.edges.items() if pid == parent_id]
        for child_id in children:
            edge_cost = self.bb.compute_distance(
                self.tree.vertices[parent_id].config,
                self.tree.vertices[child_id].config
            )
            new_cost = self.tree.vertices[parent_id].cost + edge_cost
            self.tree.vertices[child_id].set_cost(new_cost)
            self.update_children_costs(child_id)

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        if plan is None:
            return float('inf')
        return self.tree.get_vertex_for_config(plan[-1]).cost

    def extend(self, near_config, rand_config):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        sampled_goal = np.allclose(rand_config, self.goal)
        rand_config = self.goal if sampled_goal else rand_config

        if self.ext_mode == "E1":
            return rand_config

        # Unit direction vector
        direction = rand_config - near_config
        distance = np.linalg.norm(direction)

        if distance == 0:
            return None

        direction = direction / distance

        if sampled_goal and distance <= self.step_size:
            new_config = self.goal
        else:
            new_config = direction * min(self.step_size, distance) + near_config

        return new_config

    def sample_random_config(self, goal_prob, goal):
        if np.random.rand() < goal_prob:
            return goal

        while True:
            if len(self.goal) == 4:
                # REGULAR CASE
                config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
            else:
                # DOT_ENV CASE
                config = self.bb.sample_random_config(goal_prob, self.goal)
            if self.bb.config_validity_checker(config):
                return config
