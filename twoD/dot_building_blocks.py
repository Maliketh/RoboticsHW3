import itertools
import numpy as np
from shapely.geometry import Point, LineString

class DotBuildingBlocks2D(object):

    def __init__(self, env):
        self.env = env
        # robot field of fiew (FOV) for inspecting points, from [-np.pi/6, np.pi/6]
        self.ee_fov = np.pi / 3

        # visibility distance for the robot's end-effector. Farther than that, the robot won't see any points.
        self.vis_dist = 60.0

    def compute_distance(self, prev_config, next_config):
        return np.linalg.norm(np.array(next_config) - np.array(prev_config))

    def sample_random_config(self, goal_prob, goal):
        if np.random.rand() < goal_prob:
            return goal

        while True:
            # Sample a random configuration within the bounds of the environment
            x_min, x_max = self.env.xlimit[0], self.env.xlimit[1]
            y_min, y_max = self.env.ylimit[0], self.env.ylimit[1]

            x_random = np.random.uniform(x_min, x_max)
            y_random = np.random.uniform(y_min, y_max)
            config = (x_random, y_random)

            if self.config_validity_checker(config):
                return config

    def config_validity_checker(self, state):
        return self.env.config_validity_checker(state)

    def edge_validity_checker(self, state1, state2):
        return self.env.edge_validity_checker(state1, state2)
