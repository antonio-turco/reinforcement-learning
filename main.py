from tensorflow import keras
import gym
import policy_gradient
import numpy as np
import tensorflow as tf
import os
import plots.policy_gradient
import single_neuron_output

cartpole_env = gym.make("CartPole-v1")

np.random.seed(42)
tf.random.set_seed(42)
cartpole_env.seed(42)

n_inputs = cartpole_env.observation_space.shape[0]

model_filepath = "pg-cart-pole.h5"
model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid")
])

if os.path.exists(model_filepath):
    model.load_weights(model_filepath)
else:
    optimizer = keras.optimizers.Adam(lr=0.01)
    loss_fn = keras.losses.binary_crossentropy

    model, trends = policy_gradient.fit(
        cartpole_env, model, optimizer, loss_fn, 25, 10, 200, 0.95)

    model.save_weights(model_filepath)

    plots.policy_gradient.plot_reward_trend(trends, "Cart Pole", "pg-cart-pole-reward-trend.png")

single_neuron_output.render_learnt_model(cartpole_env, model, 200)
