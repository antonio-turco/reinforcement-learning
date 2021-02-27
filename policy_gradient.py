import tensorflow as tf
import numpy as np


def play_one_step(environment, observation, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(observation[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        # if action is left, then target prob is 1
        # else target prob is 0
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        # if the target is correct move the prediction to that target
        # otherwise move away from the target
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = environment.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads


def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    def episode_ended(index, max_steps, done):
        return index >= max_steps or done

    all_rewards = []
    all_gradients = []
    for _ in range(n_episodes):
        current_rewards = []
        current_gradients = []
        obs = env.reset()
        step_index = 0
        done = False
        while not episode_ended(step_index, n_max_steps, done):
            obs, reward, done, gradients = play_one_step(
                env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_gradients.append(gradients)
            step_index += 1

        all_rewards.append(current_rewards)
        all_gradients.append(current_gradients)
    return all_rewards, all_gradients


def discount_rewards(rewards, discount_factor):
    discounted_rewards = np.array(rewards)
    before_last_index = len(rewards) - 2
    for frame in range(before_last_index, -1, -1):
        discounted_rewards[frame] += discounted_rewards[frame + 1] * discount_factor
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_factor):
    all_discounted_rewards = [
        discount_rewards(rewards, discount_factor) for rewards in all_rewards
    ]

    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()

    normalized_rewards = [
        (discounted_reward - reward_mean) / reward_std
        for discounted_reward in all_discounted_rewards
    ]

    return normalized_rewards


def fit(env, model, optimizer, loss_fn, n_iterations, n_episodes_per_update, n_max_steps, discount_factor):
    metrics_trend = []
    for iteration in range(n_iterations):
        all_rewards, all_gradients = play_multiple_episodes(
            env, n_episodes_per_update, n_max_steps, model, loss_fn)

        total_rewards = sum(map(sum, all_rewards))
        current_mean_reward = total_rewards / n_episodes_per_update
        print("\rIteration: {}, mean rewards: {:.1f}".format(
            iteration, current_mean_reward), end="")
        metrics_trend.append(current_mean_reward)

        all_final_rewards = discount_and_normalize_rewards(
            all_rewards, discount_factor)
        all_mean_gradients = []
        for var_index in range(len(model.trainable_variables)):
            gradients_rewarded = [
                final_reward * all_gradients[episode_index][step][var_index]
                for episode_index, final_rewards in enumerate(all_final_rewards)
                for step, final_reward in enumerate(final_rewards)
            ]
            mean_gradients = tf.reduce_mean(gradients_rewarded, axis=0)
            all_mean_gradients.append(mean_gradients)
        optimizer.apply_gradients(
            zip(all_mean_gradients, model.trainable_variables))
    return model, metrics_trend
