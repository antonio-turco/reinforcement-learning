import numpy as np
import tensorflow as tf


class QLearning:
    def __init__(self, model, replay_buffer, n_actions, discount_factor, loss_fn):
        self.model = model
        self.replay_buffer = replay_buffer
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.loss_fn = loss_fn

    def __epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values[0])

    def __sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_buffer), size=batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def __play_one_step(self, env, state, epsilon):
        action = self.__epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = env.step(action)
        self.replay_buffer.append((state, action, reward, next_state, done))
        return next_state, reward, done, info

    def __training_step(self, batch_size, loss, optimizer):
        experiences = self.__sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards +
                           (1-dones) * self.discount_factor * max_next_Q_values)
        mask = tf.one_hot(actions, self.n_actions)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def fit(self, env, batch_size, loss, optimizer, episodes=600, steps=200):
        for episode in range(episodes):
            obs = env.reset()
            for step in range(steps):
                epsilon = max(1 - episode / 500, 0.01)
                obs, reward, done, info = self.__play_one_step(env, obs, epsilon)
                if done:
                    break

            if episode > 50:
                self.__training_step(batch_size, loss, optimizer)
