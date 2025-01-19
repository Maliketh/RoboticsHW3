import numpy as np
from RRTTree import RRTTree
import time

    
class RRTInspectionPlanner(object):
    def __init__(self, bb, start, ext_mode, goal_prob, coverage, step_size=0.5):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb, task="ip")
        self.start = start

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob
        self.coverage = coverage
        self.all_inspections = []

        self.step_size = step_size # min(self.bb.env.xlimit[-1] / 50, self.bb.env.ylimit[-1] / 200)

    def plan(self):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        # Tree sampling
        #print(f"---------------------------Tree sampling started! Goal coverage: {self.coverage}")
        iter = 0
        self.tree.add_vertex(self.start, self.bb.get_inspected_points(self.start))
        while self.coverage > self.tree.max_coverage:
            iter += 1
            rand_config = self.sample_random_config(self.goal_prob)
            near_config_id, near_config = self.tree.get_nearest_config(rand_config)
            new_config = self.extend(near_config, rand_config)

            if self.bb.edge_validity_checker(new_config, near_config):
                v = len(self.tree.vertices)
                '''
                if v % 1000 == 0:
                    print(f"Num vertices: {v}")
                    print(f"Current coverage: {self.tree.max_coverage}")
                    print(f"Num of iterations: {iter}")
                '''
                new_inspections = self.bb.get_inspected_points(new_config)
                self.all_inspections = self.bb.compute_union_of_points(new_inspections, self.all_inspections)
                inspected_pts = self.bb.compute_union_of_points(self.tree.vertices[near_config_id].inspected_points,
                                                                new_inspections)

                vid = self.tree.add_vertex(new_config, inspected_pts)
                cost = self.bb.compute_distance(new_config, near_config)
                self.tree.add_edge(near_config_id, vid, cost)
        '''
        print(f"---------------------------Tree sampling finished!"
              f"\nBest vertex: {self.tree.vertices[self.tree.max_coverage_id].config}"
              f"\nInspected points: {self.tree.vertices[self.tree.max_coverage_id].inspected_points}")
        '''
          
        # Plan computation
        plan = [self.tree.vertices[self.tree.max_coverage_id].config]
        try:
            while not np.allclose(plan[0], self.start, atol=1e-6):
                child = plan[0]
                child_idx = self.tree.get_idx_for_config(child)
                parent_idx = self.tree.edges[child_idx]
                parent = self.tree.vertices[parent_idx].config
                plan.insert(0, parent)
            return np.array(plan)
        except KeyError as e:
            print(f"No path found :(")
            return None

    def sample_random_config(self, goal_prob):
        if np.random.rand() < goal_prob:
            best_inspected_pts = self.tree.vertices[self.tree.max_coverage_id].inspected_points
            count = 0
            while count < 150:
                random_config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
                if self.bb.config_validity_checker(random_config):
                    pts_seen = self.bb.get_inspected_points(random_config)
                    if np.setdiff1d(pts_seen, best_inspected_pts).size != 0:
                        return random_config
                count += 1

        while True:
            random_config = np.random.uniform(low=-np.pi, high=np.pi, size=(4,))
            if self.bb.config_validity_checker(random_config):
                return random_config


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
        if self.ext_mode == "E1":
            return rand_config

        # Unit direction vector
        direction = rand_config - near_config
        distance = np.linalg.norm(direction)
        direction = direction / distance

        new_config = direction * self.step_size + near_config

        return new_config
