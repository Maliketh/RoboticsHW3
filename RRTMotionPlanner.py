import numpy as np
from RRTTree import RRTTree
import time

class RRTMotionPlanner(object):
    def __init__(self, bb, ext_mode, goal_prob, step_size, start, goal):
        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.start = start
        self.goal = goal

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.step_size = step_size

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        # Tree sampling
        self.tree.add_vertex(self.start)
        while not self.tree.is_goal_exists(self.goal):
            rand_config = self.sample_random_config(self.goal_prob, self.goal)  # assuming returns a valid config
            near_config_id, near_config = self.tree.get_nearest_config(rand_config)
            new_config = self.extend(near_config, rand_config)

            if self.bb.edge_validity_checker(new_config, near_config):
                vid = self.tree.add_vertex(new_config)
                cost = self.bb.compute_distance(new_config, near_config)
                self.tree.add_edge(near_config_id, vid, cost)
        print(f"Tree is generated and contains: {len(self.tree.vertices)} vertices")
        # Plan computation
        plan = [self.goal]
        try:
            while not np.allclose(plan[0], self.start, atol=1e-6):
                child = plan[0]
                child_idx = self.tree.get_idx_for_config(child)
                parent_idx = self.tree.edges[child_idx]
                parent = self.tree.vertices[parent_idx].config
                plan.insert(0, parent)
            print(np.array(plan).shape)
            return np.array(plan)
        except KeyError as e:
            print(f"No path found between {self.start} and {self.goal}")
            return None

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
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
        direction = direction / distance

        if sampled_goal and distance <= self.step_size:
            new_config = self.goal
        else:
            new_config = direction * self.step_size + near_config

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