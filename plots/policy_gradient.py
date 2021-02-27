import matplotlib.pyplot as plt


def plot_reward_trend(trends, title, filepath):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.ylabel("Mean Reward")
    plt.xlabel("Learning Iteration")
    plt.plot(trends)
    plt.savefig(filepath, transparent=True)
    plt.show()
