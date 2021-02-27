import matplotlib.pyplot as plt
from tensorflow import keras
import gym
import policy_gradient
import numpy as np
import tensorflow as tf

cartpole_env = gym.make("CartPole-v1")

np.random.seed(42)
tf.random.set_seed(42)
cartpole_env.seed(42)

n_inputs = 4

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid")
])

optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.binary_crossentropy

learnt_model, trends = policy_gradient.fit(
    cartpole_env, model, optimizer, loss_fn, 150, 10, 200, 0.95)

plt.figure(figsize=(8, 6))
plt.title("Cart Pole")
plt.ylabel("Mean Reward")
plt.xlabel("Learning Iteration")
plt.plot(trends)
plt.savefig("pg-cart-pole-reward-trend.png", transparent=True)
plt.show()
obs = cartpole_env.reset()
print(learnt_model.predict(obs[np.newaxis]))
