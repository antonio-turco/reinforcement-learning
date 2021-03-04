from tensorflow import keras
import gym
import policy_gradient
import numpy as np
import tensorflow as tf
import plots.policy_gradient
import renderer

'''
gym_name = "CartPole-v1"
experiment_name = "Cart Pole"
plot_path = "pg-cart-pole-reward-trend.png"
model_filepath = "pg-cart-pole.h5"
'''


def init(gym_name):
    environment = gym.make(gym_name)

    np.random.seed(42)
    tf.random.set_seed(42)
    environment.seed(42)

    n_inputs = environment.observation_space.shape[0]

    model = keras.models.Sequential([
        keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
        keras.layers.Dense(environment.action_space.n, activation="softmax")
    ])

    return environment, model


def train_model(gym_name, experiment_name, plot_path, model_filepath):
    environment, model = init(gym_name)
    optimizer = keras.optimizers.Adam(lr=0.01)
    loss_fn = keras.losses.binary_crossentropy

    model, trends = policy_gradient.fit(
        environment, model, optimizer, loss_fn, 50, 15, 200, 0.99999)

    model.save_weights(model_filepath)

    plots.policy_gradient.plot_reward_trend(
        trends, experiment_name, plot_path)


def run_simulation(gym_name, model_filepath):
    env, model = init(gym_name)
    model.load_weights(model_filepath)
    renderer.play_model(env, model, 200)


train_model("MountainCar-v0", "Mountain Car", "pg-mountain-reward-trend.png", "pg-mountain.h5")
run_simulation("MountainCar-v0", "pg-mountain.h5")
