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
        """
        Compute the Euclidean distance between two configurations.
        @param prev_config Previous configuration.
        @param next_config Next configuration.
        """
        return np.linalg.norm(np.array(next_config) - np.array(prev_config))

    def sample_random_config(self, goal_prob, goal):
        """
        Samples a random configuration in the environment with a certain probability of returning the goal.
        Ensures the sampled configuration is collision-free.

        Args:
            goal_prob (float): Probability of sampling the goal configuration.
            goal (tuple): The goal configuration as (x, y).

        Returns:
            tuple: A randomly sampled configuration (x, y) that is collision-free.
        """
        while True:
            if np.random.rand() < goal_prob:
                # With probability `goal_prob`, return the goal configuration
                config = goal
            else:
                # Sample a random configuration within the bounds of the environment
                x_min, x_max = self.x_bounds
                y_min, y_max = self.y_bounds

                x_random = np.random.uniform(x_min, x_max)
                y_random = np.random.uniform(y_min, y_max)
                config = (x_random, y_random)

            # Check if the configuration is collision-free
            if self.is_collision_free(config):
                return config

    def config_validity_checker(self, state):
        return self.env.config_validity_checker(state)

    def edge_validity_checker(self, state1, state2):
        return self.env.edge_validity_checker(state1, state2)


