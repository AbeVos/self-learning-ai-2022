import random

from collections import defaultdict

import gym
import matplotlib.pyplot as plt

from cliffwalking import CliffWalking
from plot_mc_prediction import create_plots

# Monte Carlo Prediction


def simple_policy(state):
    # SCHRIJF JE CODE HIER
    hand_sum, _, _ = state

    if hand_sum < 20:
        return 1
    else:
        return 0


def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    # Keep track of how many times each state has been visited.
    # A defaultdict creates a default value (0) when we want to read a key
    # that is not stored, e.g.: `print(state_count["some state"])` prints `0`.
    N = defaultdict(int)

    # The final value function
    V = defaultdict(float)

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        episode = []

        while not done:
            action = policy(state)
            state_new, reward, done, _ = env.step(action)

            # JOUW CODE HIER
            # Sla informatie van deze stap op in een tuple en voeg die toe
            # aan de lijst.
            episode.append((state, reward))

            state = state_new

        G = 0

        for step in episode[::-1]:  # Loop through the steps backward.
            state, reward = step  # JOUW CODE HIER: Pak de tuple uit.

            # JOUW CODE HIER
            # Update G, N en V volgens het MC prediction algoritme
            G = discount_factor * G + reward

            N[state] += 1
            V[state] += (G - V[state]) / N[state]

    return V


env = gym.make("Blackjack-v1")
create_plots(env, simple_policy, mc_prediction)

# Sarsa & Q-learning


def get_optimal_value(Q, s, nA):
    action_values = Q.get(s, [0 for _ in range(nA)])
    return max(action_values)


def egreedy_policy(Q, s, nA, epsilon=0.1):
    if s not in Q:
        Q[s] = list(range(nA))

    if random.random() < epsilon:
        return random.randint(0, nA - 1)
    else:
        return max(
            list(range(nA)),
            key=lambda a: Q[s][a],
        )


def sarsa(env, num_episodes, learning_rate=0.5, discount_factor=1.0):
    Q = {}
    nA = env.action_space.n
    rewards = []

    for episode in range(num_episodes):
        done = False
        total_reward = 0

        state = env.reset()
        action = egreedy_policy(Q, state, nA)

        while not done:
            state_new, reward, done, _ = env.step(action)
            total_reward += reward

            # JOUW CODE HIER
            # Kies een nieuwe actie met het egreedy beleid.

            action_new = egreedy_policy(Q, state_new, nA)

            # Update `Q[state][action]`
            Q[state][action] += learning_rate * (
                reward
                + discount_factor * Q[state_new][action_new]
                - Q[state][action]
            )

            # Update `state` en `action` voor de volgende iteratie.
            state = state_new
            action = action_new

        rewards.append(total_reward)

        print(f"Episode {episode}, sum reward: {total_reward}")

    return Q, rewards


def q_learning(env, num_episodes, learning_rate=0.5, discount_factor=1.0):
    Q = {}
    nA = env.action_space.n
    rewards = []

    for episode in range(num_episodes):
        done = False
        total_reward = 0

        state = env.reset()

        while not done:
            # JOUW CODE HIER
            # Kies een nieuwe actie met het egreedy beleid.
            action = egreedy_policy(Q, state, nA)

            state_new, reward, done, _ = env.step(action)
            total_reward += reward

            # JOUW CODE HIER
            # Update `Q[state][action]`, gebruik `get_optimal_value`.
            Q[state][action] += learning_rate * (
                reward
                + discount_factor * get_optimal_value(Q, state_new, nA)
                - Q[state][action]
            )

            # Update `state` voor de volgende iteratie.
            state = state_new

        rewards.append(total_reward)

        print(f"Episode {episode}, sum reward: {total_reward}")

    return Q, rewards


env = CliffWalking()

Q_sarsa, rewards_sarsa = sarsa(env, 500)
Q_qlearning, rewards_qlearning = q_learning(env, 500)

plt.plot(rewards_sarsa, label="Sarsa")
plt.plot(rewards_qlearning, label="Q-learning")
plt.legend()
plt.show()
