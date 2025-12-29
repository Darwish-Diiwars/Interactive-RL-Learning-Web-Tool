import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class StochasticGridWorld:
    def __init__(self, size=6):
        self.size = size
        self.state = (0, 0)
        self.goal = (5, 5)
        self.obstacles = [(1,2), (2,3), (3,1), (4,4)]
        self.actions = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}  # Up, Down, Left, Right
        self.slip_prob = 0.1

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        if random.random() < self.slip_prob:
            action = (action + random.choice([-1, 1])) % 4
        dx, dy = self.actions[action]
        new_x = max(0, min(self.size-1, self.state[0] + dx))
        new_y = max(0, min(self.size-1, self.state[1] + dy))
        if (new_x, new_y) in self.obstacles:
            reward = -10
            done = False
        elif (new_x, new_y) == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        self.state = (new_x, new_y) if (new_x, new_y) not in self.obstacles else self.state
        return self.state, reward, done, {}

    def all_states(self):
        return [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) != self.goal]

    def transitions(self, s, a):
        self.state = s
        next_s, r, done, _ = self.step(a)
        return [(next_s, r, 1.0)]

    def render(self):
        grid = np.zeros((self.size, self.size))
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = -1
        grid[self.goal[0], self.goal[1]] = 2
        grid[self.state[0], self.state[1]] = 1
        return grid

class TaxiWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = range(self.env.action_space.n)

    def reset(self):
        s, _ = super().reset()
        return s  # Integer state

    def step(self, a):
        next_s, r, done, _, _ = super().step(a)
        return next_s, r, done, {}

    def all_states(self):
        return list(range(self.env.observation_space.n))  # 0 to 499

    def transitions(self, s, a):
        trans = self.unwrapped.P[s][a]
        return [(next_s, r, p) for p, next_s, r, done in trans]

    def render(self):
        return self.env.render()

class CartPoleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.bins = 10
        self.actions = range(self.env.action_space.n)

    def reset(self):
        s, _ = super().reset()
        return self.to_discrete(s)

    def step(self, a):
        next_s, r, done, _, _ = super().step(a)
        return self.to_discrete(next_s), r, done, {}

    def to_discrete(self, s):
        pos_bin = min(self.bins - 1, max(0, int((s[0] + 2.4) / 4.8 * self.bins)))
        vel_bin = min(self.bins - 1, max(0, int((s[1] + 3) / 6 * self.bins)))
        angle_bin = min(self.bins - 1, max(0, int((s[2] + 0.209) / 0.418 * self.bins)))
        ang_vel_bin = min(self.bins - 1, max(0, int((s[3] + 3) / 6 * self.bins)))
        return (pos_bin, vel_bin, angle_bin, ang_vel_bin)

    def all_states(self):
        return [(i, j, k, l) for i in range(self.bins) for j in range(self.bins) for k in range(self.bins) for l in range(self.bins)]

    def transitions(self, s, a):
        raise NotImplementedError("Model-based algos not supported for CartPole-v1")

    def render(self):
        return self.env.render()

def get_env(env_name):
    if env_name == "StochasticGridWorld":
        return StochasticGridWorld()
    elif env_name == "Taxi-v3":
        env = gym.make(env_name, render_mode="rgb_array")
        return TaxiWrapper(env)
    elif env_name == "CartPole-v1":
        env = gym.make(env_name, render_mode="rgb_array")
        return CartPoleWrapper(env)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def visualize_env(env, state, env_name):
    if env_name == "StochasticGridWorld":
        grid = env.render()
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='gray')
        ax.set_title("Environment State")
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')
    elif env_name == "Taxi-v3":
        rgb = env.render()
        fig, ax = plt.subplots()
        ax.imshow(rgb)
        ax.set_title("Environment State (Taxi)")
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')
    else:
        rgb = env.render()
        fig, ax = plt.subplots()
        ax.imshow(rgb)
        ax.set_title("Environment State")
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')

def visualize_values(V, env, env_name):
    if env_name == "StochasticGridWorld":
        size = env.size
        grid = np.zeros((size, size))
        for s, val in V.items():
            grid[s[0], s[1]] = val
        fig, ax = plt.subplots()
        im = ax.imshow(grid, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Value')
        ax.set_title("Value Function")
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')
    elif env_name == "Taxi-v3":
        # Taxi has 500 states – show text
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Value Function\nLearned {len(V)} states", ha='center')
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Value Function (discretized)\nLearned {len(V)} states", ha='center')
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')

def visualize_policy(policy, env, env_name):
    if env_name == "StochasticGridWorld":
        size = env.size
        fig, ax = plt.subplots()
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.invert_yaxis()
        arrows = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
        for s, a in policy.items():
            if isinstance(a, dict):
                a = max(a, key=a.get)
            dx, dy = arrows.get(a, (0,0))
            ax.arrow(s[1] + 0.5, s[0] + 0.5, dx, dy, head_width=0.2, color='red')
        ax.set_title("Policy (red arrows = best action)")
        ax.grid(True)
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')
    elif env_name == "Taxi-v3":
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Policy learned\n(500 states – arrows not shown)", ha='center')
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')
    else:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Policy (discretized)\nArrows not shown for this env", ha='center')
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.read()).decode('utf-8')