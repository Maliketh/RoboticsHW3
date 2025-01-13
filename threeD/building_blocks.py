import numpy as np
import math

class BuildingBlocks3D(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, transform, ur_params, env, resolution=0.1, p_bias=0.05):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07, 0.05])

        self.single_mechanical_limit = list(self.ur_params.mechamical_limits.values())[-1][-1]

        # pairs of links that can collide during sampling
        self.possible_link_collisions = [['shoulder_link', 'forearm_link'],
                                         ['shoulder_link', 'wrist_1_link'],
                                         ['shoulder_link', 'wrist_2_link'],
                                         ['shoulder_link', 'wrist_3_link'],
                                         ['upper_arm_link', 'wrist_1_link'],
                                         ['upper_arm_link', 'wrist_2_link'],
                                         ['upper_arm_link', 'wrist_3_link'],
                                         ['forearm_link', 'wrist_2_link'],
                                         ['forearm_link', 'wrist_3_link']]

    def sample_random_config(self, goal_prob,  goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        :param goal_prob - the probability that goal should be sampled
        """
        # TODO: HW2 5.2.1
        pass

    def config_validity_checker(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration
        """
        # Transform configuration to global sphere coordinates
        global_sphere_coords = self.transform.conf2sphere_coords(conf)

        # Check for collisions with the floor (z = 0)
        for i, link in enumerate(global_sphere_coords.keys()):
            for j, sphere in enumerate(global_sphere_coords[link]):
                if i != 0:  # or j != 0:  #skip the base link
                    if sphere[2] - self.ur_params.sphere_radius[link] < 0:
                        # print("COLLISION WITH FLOOR DETECTED")
                        return True  # Floor collision detected

        # Check for internal collisions
        for link1, link2 in self.possible_link_collisions:
            if link1 not in global_sphere_coords or link2 not in global_sphere_coords:
                continue

            for sphere1 in global_sphere_coords[link1]:
                for sphere2 in global_sphere_coords[link2]:
                    dist = np.linalg.norm(sphere1 - sphere2)
                    if dist < self.ur_params.sphere_radius[link1] + self.ur_params.sphere_radius[link2]:
                        # print("INTERNAL COLLISION DETECTED")
                        return True  # Internal collision detected
            # Check if the manipulator exceeds the x-direction limit (0.4 m)
        for link in global_sphere_coords.keys():
            for sphere in global_sphere_coords[link]:
                if sphere[0] > 0.4:
                    # Manipulator exceeds x-direction limit
                    return False

        if len(self.env.obstacles) != 0:
            # Check for collisions with obstacles
            for link in global_sphere_coords.keys():
                for sphere in global_sphere_coords[link]:
                    for obstacle in self.env.obstacles:
                        dist = np.linalg.norm(sphere - obstacle)
                        if dist < self.ur_params.sphere_radius[link] + self.env.radius:
                            # print("OBSTACLE COLLISION DETECTED")
                            return True  # Obstacle collision detected

        # No collisions detected
        return False


    def edge_validity_checker(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        angular_differences = [abs(current - prev) for current, prev in zip(current_conf, prev_conf)]
        num_configs = max(3, math.ceil(max(angular_differences) / self.resolution))
        configs = np.linspace(prev_conf, current_conf, num_configs)
        for conf in configs:
            if self.config_validity_checker(conf):
                return False
        return True
    def compute_distance(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1 - conf2, 2)) ** 0.5
