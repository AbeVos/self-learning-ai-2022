import numpy as np
import random

from gym import Env, spaces
from itertools import product

MAX_X = 12
MAX_Y = 4
N_STATES = MAX_X * MAX_Y
N_ACTIONS = 4

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class CliffWalking(Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)

        P = {}

        for x, y in product(range(MAX_X), range(MAX_Y)):
            state = y * MAX_X + x
            P[state] = {}

            # Get the next state given an action, taking the edges
            # of the board into account.
            ns_up = state if y == 0 else state - MAX_X
            ns_right = state if x == (MAX_X - 1) else state + 1
            ns_down = state if y == (MAX_Y - 1) else state + MAX_X
            ns_left = state if x == 0 else state - 1

            # Create the dynamics for each action in this state.
            P[state][UP] = [(1.0, ns_up, -1, False)]
            P[state][RIGHT] = [(1.0, ns_right, -1, False)]
            P[state][DOWN] = [(1.0, ns_down, -1, False)]
            P[state][LEFT] = [(1.0, ns_left, -1, False)]

            if y == 1 and x in list(range(1, MAX_X - 2)):
                # If we walk into the cliff, we lose 100 points and
                # move back to start.
                P[state][UP] = [(1.0, 0, -100, False)]
            elif y == 0 and x == 0:
                P[state][RIGHT] = [(1.0, 0, -100, False)]
            elif y == 0 and x == MAX_X - 1:
                for action in range(N_ACTIONS):
                    P[state][action] = [(1.0, state, 0, True)]

        self.P = P

    def reset(self):
        self.current_state = 0

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
        row1 = ["   " for _ in range(MAX_X)]
        row1[0] = " S "
        row1[MAX_X - 1] = " G "
        row1[1 : MAX_X - 1] = (MAX_X - 2) * [" # "]

        rows = [row1] + [MAX_X * ["   "] for _ in range(MAX_Y - 1)]

        x = self.current_state % MAX_X
        y = (self.current_state - x) // MAX_X

        rows[y][x] = " O "
        rows = rows[::-1]

        print(
            ("\n" + MAX_X * "---+" + "\n").join(
                ["|".join(row) for row in rows]
            ),
            "\n",
        )
