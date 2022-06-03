import math
import random

import gym
import matplotlib.pyplot as plt
import numpy as np

LL_LOW = np.array(
    [
        # these are bounds for position
        # realistically the environment should have ended
        # long before we reach more than 50% outside
        -1.5,
        -1.5,
        # velocity bounds is 5x rated speed
        -5.0,
        -5.0,
        -math.pi,
        -5.0,
        -0.0,
        -0.0,
    ]
).astype(np.float32)
LL_HIGH = np.array(
    [
        # these are bounds for position
        # realistically the environment should have ended
        # long before we reach more than 50% outside
        1.5,
        1.5,
        # velocity bounds is 5x rated speed
        5.0,
        5.0,
        math.pi,
        5.0,
        1.0,
        1.0,
    ]
).astype(np.float32)


# Episodic semi-gradient Sarsa


def q_value(state, action, params):
    return state @ params[action]


def egreedy_policy(params, state, epsilon):
    n_actions = len(params)

    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return max(
            list(range(n_actions)),
            key=lambda a: q_value(state, a, params),
        )


def semigrad_sarsa(
    env, num_episodes, learning_rate=1e-2, discount_factor=1.0, epsilon=0.1
):
    nS = env.observation_space.shape[0]
    nA = env.action_space.n
    params = np.zeros((nA, nS))
    rewards = []

    for episode in range(num_episodes):
        done = False
        total_reward = 0

        state = env.reset()
        action = egreedy_policy(params, state, epsilon)

        while not done:
            state_new, reward, done, _ = env.step(action)
            total_reward += reward

            # JOUW CODE HIER
            # Kies een nieuwe actie met het egreedy beleid.
            action_new = egreedy_policy(params, state_new, epsilon)

            # Update `params` voor actie `action`.
            if done:
                params[action] += (
                    learning_rate
                    * (reward - q_value(state, action, params))
                    * state
                )
                continue

            params[action] += (
                learning_rate
                * (
                    reward
                    + discount_factor * q_value(state_new, action_new, params)
                    - q_value(state, action, params)
                )
                * state
            )

            # Update `state` end `action` voor de volgende iteratie.
            state = state_new
            action = action_new

        rewards.append(total_reward)

        print(f"Episode {episode}, sum reward: {total_reward}")

    return params, rewards


def evaluate(env, params, render=False):
    state = env.reset()

    done = False
    total_reward = 0

    while not done:
        action = egreedy_policy(params, state, 0)
        state, reward, done, _ = env.step(action)

        total_reward += reward

        if render:
            env.render()

    return total_reward


"""
env = gym.make("CartPole-v1")
nS = env.observation_space.shape[0]
nA = env.action_space.n
params = np.zeros((nA, nS))

reward_before = sum([evaluate(env, params) for _ in range(100)]) / 100

params, rewards = semigrad_sarsa(env, 5, learning_rate=1e-2, epsilon=0.2)

reward_after = sum([evaluate(env, params) for _ in range(100)]) / 100

evaluate(env, params, True)
env.close()

print("Reward before:", reward_before, "after:", reward_after)

# plt.plot(rewards)
# plt.show()

#
"""


def policy(state, params):
    # JOUW CODE HIER
    # Vermenigvuldig `state` en `params`, let op dat de volgorde belangrijk is
    # bij matrix-vermenigvuldiging.
    # Vind de index van de actie met de hoogste score.
    # state = (state - LL_LOW) / (LL_HIGH - LL_LOW)
    return np.argmax(params @ state)


def rollout(env, params, render=False):
    state = env.reset()

    done = False
    total_reward = 0

    while not done:
        action = policy(state, params)

        state, reward, done, _ = env.step(action)

        total_reward += reward

        if render:
            env.render()

    return total_reward


def brs(env, num_iters, lr=0.2, v=0.2, N=4):
    nS = env.observation_space.shape[0]
    nA = env.action_space.n
    params = np.zeros((nA, nS))

    for iteration in range(num_iters):
        # JOUW CODE HIER.
        results = []

        for n in range(N):
            # Genereer de noise vectors.
            noise = np.random.randn(*params.shape)

            # Genereer de twee beloningen (met de `rollout` functie) voor elke
            # noise vector en sla die samen met de bijbehorende noise vector op
            # in een lijst.
            reward_plus = rollout(env, params + v * noise)
            reward_min = rollout(env, params - v * noise)

            results.append((noise, reward_plus, reward_min))

        # Update de parameters.
        step = np.mean(
            [
                noise * (reward_plus - reward_min)
                for noise, reward_plus, reward_min in results
            ],
            axis=0,
        )
        params += lr * step

        print(f"It: {iteration}, reward: {rollout(env, params)}")

    return params


def argmaxmulti(array, b=5):
    return np.argpartition(-np.array(array), b)[:b]


def ars(env, num_iters, lr=0.2, v=0.2, N=4, b=2):
    nS = env.observation_space.shape[0]
    nA = env.action_space.n
    params = np.zeros((nA, nS))

    for iteration in range(num_iters):
        # JOUW CODE HIER.
        results = []
        max_rewards = []

        # Genereer de noise vectors.
        for n in range(N):
            noise = np.random.randn(*params.shape)

            # Genereer de twee beloningen (met de `rollout` functie) voor elke
            # noise vector en sla die samen met de bijbehorende noise vector op
            # in een lijst.
            reward_plus = rollout(env, params + v * noise)
            reward_min = rollout(env, params - v * noise)

            results.append((noise, reward_plus, reward_min))
            max_rewards.append(max(reward_plus, reward_min))

        # print("Before", results)
        results = [results[idx] for idx in sorted(argmaxmulti(max_rewards))]
        # print("After", results)
        reward_sd = np.std([(plus, minus) for _, plus, minus in results])

        # Update de parameters.
        step = np.mean(
            [
                noise * (reward_plus - reward_min)
                for noise, reward_plus, reward_min in results
            ],
            axis=0,
        )
        params += lr / reward_sd * step

        print(f"It: {iteration}, reward: {rollout(env, params)}")

    return params


env = gym.make("LunarLander-v2")

nS = env.observation_space.shape[0]
nA = env.action_space.n
params = np.zeros((nA, nS))
reward_before = sum([evaluate(env, params) for _ in range(100)]) / 100

params = brs(env, 100, lr=1e-3, N=10)
print(params)

reward_after = sum([evaluate(env, params) for _ in range(100)]) / 100

print("Reward before:", reward_before, "after:", reward_after)

rollout(env, params, True)

"""
nS = env.observation_space.shape[0]
nA = env.action_space.n
params = np.zeros((nA, nS))
reward_before = sum([evaluate(env, params) for _ in range(100)]) / 100

params = ars(env, 100, lr=1e-3, N=10, b=10)

reward_after = sum([evaluate(env, params) for _ in range(100)]) / 100

print("Reward before:", reward_before, "after:", reward_after)

rollout(env, params, True)
env.close()
"""
