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

    def plan2(self):
        self.tree.add_vertex(self.start)
        improvement_counter = 0
        path_costs = []
        for i in range(self.max_itr):
            if self.tree.is_goal_exists(self.goal) and (i - improvement_counter >= 30 or i == 0):
                improvement_counter = i
                path_costs.append((self.tree.get_vertex_for_config(self.goal).cost, i))

            if self.stop_on_goal and self.tree.is_goal_exists(self.goal):
                break

            rand_config = self.bb.sample_random_config(self.goal_prob, self.goal)

            near_config_id, near_config = self.tree.get_nearest_config(rand_config)
            new_config = self.extend(near_config, rand_config)
            if new_config is not None:
                if self.bb.edge_validity_checker(new_config, near_config):
                    vid = self.tree.add_vertex(new_config)
                    cost = self.bb.compute_distance(new_config, near_config)
                    self.tree.add_edge(near_config_id, vid, cost)

                    near_ids, near_configs = self.tree.get_k_nearest_neighbors(new_config,
                                                                               min(self.k, len(self.tree.vertices)-1))

                    # rewire(near_config, new_config)
                    for idx, near_id in enumerate(near_ids):
                        self.rewire(near_configs[idx], near_ids[idx], new_config, vid)

                    # rewire(new_config, near_config)
                    for idx, near_id in enumerate(near_ids):
                        self.rewire(new_config, vid, near_configs[idx], near_ids[idx])

        goal_idx = self.tree.get_idx_for_config(self.goal)
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

            return np.array(plan), path_costs
        except Exception as e:
            return None, None

    def rewire(self, parent_config, parent_id, child_config, child_id):
        if self.bb.edge_validity_checker(parent_config, child_config):
            edge_cost = self.bb.compute_distance(parent_config, child_config)
            total_cost = self.tree.vertices[parent_id].cost + edge_cost
            if total_cost < self.tree.vertices[child_id].cost:
                self.tree.add_edge(parent_id, child_id, edge_cost)
                self.update_children_costs(child_id)

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
            #print("Extended goal")
            new_config = self.goal
        else:
            new_config = direction * min(self.step_size, distance) + near_config

        return new_config
