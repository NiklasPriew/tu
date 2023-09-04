import pickle
from collections import defaultdict

import numpy as np

from game import Breakout

def create_layout(grid_height, grid_width, grid_type='single'):
    assert (0 <= grid_height <= 10) and (0 <= grid_width <= 15) and (grid_width % 3 == 0)
    assert grid_type in ['single', 'double', 'bridge', 'towers']

    grid_layout = np.zeros((grid_height, grid_width // 3), dtype=np.int8)
    line_layout = grid_layout.copy()

    for i in range(grid_width // 3):
        # first row from top
        if grid_type in ['single', 'double', 'bridge']:
            line_layout[grid_height - 1, i] = 1
        else:  # 'towers'
            line_layout[grid_height - 1, i] = 1 if i % 2 == 0 else 0

        # second row from top
        if grid_type in ['double']:
            line_layout[grid_height - 2, i] = 1
        elif grid_type in ['bridge', 'towers']:
            line_layout[grid_height - 2, i] = 1 if i % 2 == 0 else 0

    return line_layout

class MonteCarloControl:
    def __init__(self, layout, grid_height: int, grid_width: int, grid_type: str, epsilon: float, gamma: float):
        self.layout = layout
        self.layout_copy = layout.copy()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid_type = grid_type
        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = np.asarray([-1, 0, 1], dtype=np.int8)
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.reward_history = []
        self.rng = np.random.default_rng(0)
        self.state_ball_paddle = None
        self.state_bricks = None

    # @profile
    def select_action_soft(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        else:
            q_values = [self.Q[state, a] for a in self.actions]
            return self.rng.choice(np.flatnonzero(q_values == np.max(q_values))) - 1

    # @profile
    def select_action_greedy(self, state):
        q_values = [self.Q[state, a] for a in self.actions]
        return np.argmax(q_values) - 1

    def move(self, breakout, policy, episode_memory, n_steps):
        n_steps += 1
        state = np.append(self.state_ball_paddle, self.state_bricks).tobytes()
        action = policy(state)
        reset, reward, self.state_ball_paddle, self.state_bricks, game_status, won = breakout.update(action)
        episode_memory.append([state, action, reward])

        return reset, game_status, episode_memory, won, n_steps

    def play(self, policy, ball_x_speed_start=None):
        self.state_ball_paddle = np.asarray([self.grid_width // 2 + self.rng.integers(low=-2, high=2, size=1)[0], 1,
                                             self.rng.integers(low=-2, high=2, size=1)[0] if ball_x_speed_start is None else ball_x_speed_start, 1,
                                             self.grid_width // 2 - 2, 0],
                                            dtype=np.int8)
        self.state_bricks = self.layout.ravel()
        breakout = Breakout(self.layout.copy(),
                            self.state_ball_paddle[0], self.state_ball_paddle[1],
                            self.state_ball_paddle[2], self.state_ball_paddle[3],
                            self.state_ball_paddle[4], self.state_ball_paddle[5],
                            self.grid_height, self.grid_width, self.grid_type)

        # Generate episodes according to policy
        episode_memory = []
        n_steps = 0  # number of steps in current episode
        game_status = True
        while game_status:
            reset, game_status, episode_memory, won, n_steps = self.move(breakout, policy, episode_memory, n_steps)
        episode_memory.append([np.append(self.state_ball_paddle, self.state_bricks).tobytes(), 0, 0])

        return episode_memory, won, n_steps

    # @profile
    def learn(self, n_episodes, greedy=False):
        policy = self.select_action_soft  # [self.select_action_greedy, self.select_action_soft]

        wins = np.zeros(n_episodes)
        steps = np.zeros(n_episodes)
        for i in range(n_episodes):
            episode_memory, won, n_steps = self.play(policy)
            wins[i] = won
            steps[i] = n_steps

            # Learn/update policy from episode memory
            memap = defaultdict()
            for t in range(len(episode_memory) - 1, -1, -1):
                (Sti, Ati, Rtj) = episode_memory[t]
                memap[Sti, Ati] = (Sti, Ati, Rtj, t)

            if not greedy:
                G = 0
                total_reward = 0
                for t in range(len(episode_memory) - 1, -1, -1):
                    (Sti, Ati, Rtj) = episode_memory[t]
                    G = self.gamma * G + Rtj
                    total_reward += Rtj
                    if (Sti, Ati) not in memap or t <= memap[Sti, Ati][3]:
                        self.N[Sti, Ati] = self.N[Sti, Ati] + 1
                        self.Q[Sti, Ati] = (self.Q[Sti, Ati] * (self.N[Sti, Ati] - 1) + G) / self.N[Sti, Ati]
                self.reward_history.append(total_reward)

        return wins, steps

    def load_model(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.Q = data[0]
        self.N = data[1]

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump([self.Q, self.N], f)
