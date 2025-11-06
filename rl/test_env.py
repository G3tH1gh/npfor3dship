import gym
from gym import spaces
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class Env(gym.Env):

    def __init__(self, seq_length=5 , num_targets=16):
        super(Env, self).__init__()

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(3,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(seq_length, 6),
            dtype=np.float32
        )

        self.seq_length = seq_length
        self.num_targets = num_targets
        self.max_steps = 50
        self.steps = 0


        self.state = np.zeros(3)
        self.targets = []
        self.visited = np.zeros(16, dtype=bool)
        self.path = None
        self.current_target_idx = 0
        self.steps_at_current_target = 0
        self.total_steps = 0
        self.reach_reward = 1
        self.shorter_reward = 10
        self.total_reward = 0

    def reset(self):
        self.state = np.zeros(3)
        self.visited = np.zeros(16, dtype=bool)
        self.targets = []
        self.current_target_idx = 0
        self.steps = 0
        self.path = None
        self._solve_tsp()
        self.current_target_idx =
        return self.state

    def step(self, action):
        displacement = action * 5.0
        self.state = self.state + displacement
        self.steps += 1
        reward = 0
        done = False

        current_target = self.targets[self.current_target_idx]
        distance = np.linalg.norm(self.state - current_target)
        if distance < self.distance_threshold:
            target_id = self.current_target_idx
            self.visited[target_id] = True
            reward += self.reach_reward

            if not np.all(self.visited):
                self._solve_tsp()
                self.current_target_idx = 0
            else:
                reward += self.complete_reward
                done = True

        if self.steps >= self.max_steps:
            done = True

        current_target = self.path[self.current_target_idx]
        self.total_reward += reward
        return self.state, reward, done

    def _solve_tsp(self):

        unvisited = self.targets[~self.visited]

        if len(unvisited) == 0:
            self.path = np.array([self.agent_pos])
            return

        points = np.vstack([self.agent_pos, unvisited])
        dist_matrix = cdist(points, points)

        row_idx, col_idx = linear_sum_assignment(dist_matrix)


        path_indices = [0]
        current_idx = 0

        for _ in range(len(points) - 1):
            next_idx = col_idx[current_idx]
            path_indices.append(next_idx)
            current_idx = next_idx

        self.path = points[path_indices][1:]

    def render(self, mode='human'):
        print(f'Current state: {self.state},Total Reward: {self.total_reward}')

    def close(self):
