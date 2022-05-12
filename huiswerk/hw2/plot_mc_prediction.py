import matplotlib.pyplot as plt
import numpy as np


def make_plottable(V):
    data = np.zeros((2, 10, 10))

    for (hand, dealer, usable), reward in V.items():
        data[int(usable), hand - 12, dealer - 1] = reward

    return data


def plot_V(data, usable, xticks, yticks, binary=False, show_label=False):
    if binary:
        plt.imshow(data[usable, ::-1], cmap="gray")
    else:
        plt.imshow(data[usable, ::-1], vmin=-1, vmax=1)
        plt.colorbar()
    plt.xticks(np.arange(0, 10), xticks)
    plt.yticks(np.arange(0, 10), yticks)

    if show_label:
        plt.xlabel("Dealer showing")
        plt.ylabel("Player sum")


def create_plots(env, policy, func):
    print("Running MC Prediction with 10k episodes")
    V_10k = func(policy, env, num_episodes=10000)
    print("Running MC Prediction with 500k episodes")
    V_500k = func(policy, env, num_episodes=500000)

    data_10k = make_plottable(V_10k)
    data_500k = make_plottable(V_500k)

    xticks = list(range(1, 11))
    xticks[0] = "A"

    yticks = list(range(12, 22))[::-1]

    plt.plot()
    plt.subplot(221)
    plot_V(data_10k, 0, xticks, yticks)
    plt.title("10k, no usable ace")

    plt.subplot(222)
    plot_V(data_500k, 0, xticks, yticks)
    plt.title("500k, no usable ace")

    plt.subplot(223)
    plot_V(data_10k, 1, xticks, yticks, show_label=True)
    plt.title("10k, usable ace")

    plt.subplot(224)
    plot_V(data_500k, 1, xticks, yticks, show_label=True)
    plt.title("500k, usable ace")

    plt.tight_layout()
    plt.show()
