import matplotlib.pyplot as plt
from tensorflow import keras
import gym
import policy_gradient
import numpy as np
import tensorflow as tf
import os
import random


def plot_reward_trend(trends, title, filepath):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.ylabel("Mean Reward")
    plt.xlabel("Learning Iteration")
    plt.plot(trends)
    plt.savefig(filepath, transparent=True)
    plt.show()


def render_learnt_model(env, model, steps):
    obs = env.reset()
    for _ in range(steps):
        env.render()
        left_prob = model.predict(obs[np.newaxis])
        action = random.uniform(0, 1) > left_prob
        action = int(action)
        obs, reward, done, info = cartpole_env.step(action)
        if done:
            break


cartpole_env = gym.make("CartPole-v1")

np.random.seed(42)
tf.random.set_seed(42)
cartpole_env.seed(42)
random.seed(42)

n_inputs = cartpole_env.observation_space.shape[0]

model_filepath = "pg-cart-pole.h5"
model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid")
])

if os.path.exists(model_filepath):
    model.load_weights(model_filepath)
    render_learnt_model(cartpole_env, model, 200)
else:
    optimizer = keras.optimizers.Adam(lr=0.01)
    loss_fn = keras.losses.binary_crossentropy

    learnt_model, trends = policy_gradient.fit(
        cartpole_env, model, optimizer, loss_fn, 150, 10, 200, 0.95)

    learnt_model.save_weights(model_filepath)

    plot_reward_trend(trends, "Cart Pole", "pg-cart-pole-reward-trend.png")
