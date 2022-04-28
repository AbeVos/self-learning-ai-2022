import numpy as np
import random

from gym import Env, spaces
from itertools import product

MAX_X = 5
MAX_Y = 5
N_STATES = MAX_X * MAX_Y
N_ACTIONS = 4

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class Gridworld(Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)

        P = {}

        for x, y in product(range(MAX_X), range(MAX_Y)):
            state = y * MAX_X + x
            P[state] = {}

            if state == 1:
                for action in range(N_ACTIONS):
                    P[state][action] = [(1.0, 21, 10, False)]
            elif state == 3:
                for action in range(N_ACTIONS):
                    P[state][action] = [(1.0, 13, 5, False)]
            else:
                ns_up = state if y == 0 else state - MAX_X
                ns_right = state if x == (MAX_X - 1) else state + 1
                ns_down = state if y == (MAX_Y - 1) else state + MAX_X
                ns_left = state if x == 0 else state - 1

                P[state][UP] = [(1.0, ns_up, -(ns_up == state), False)]
                P[state][RIGHT] = [
                    (1.0, ns_right, -(ns_right == state), False)
                ]
                P[state][DOWN] = [(1.0, ns_down, -(ns_down == state), False)]
                P[state][LEFT] = [(1.0, ns_left, -(ns_left == state), False)]

        self.P = P
        self.state_dist = np.ones(N_STATES) / N_STATES

    def reset(self):
        self.current_state = random.randint(0, N_STATES)

        return self.current_state

    def step(self, action):
        transitions = self.P[self.current_state][action]
        probs = [trans[0] for trans in transitions]

        ((_, next_state, reward, done),) = random.choices(
            population=transitions,
            weights=probs,
            k=1,
        )

        self.current_state = next_state

        return next_state, reward, done, {}

    def render(self):
        rows = [
            " A B ",
            " : v ",
            " : B ",
            " v   ",
            " A   ",
        ]

        for y, row in enumerate(rows):
            for x in range(MAX_X):
                state = y * MAX_X + x
                if state == self.current_state:
                    row = list(row)
                    row[x] = "O"
                    break

            rows[y] = " " + " | ".join(row) + " \n"

        sep = "+".join(["---"] * MAX_X) + "\n"
        return sep.join(rows)
