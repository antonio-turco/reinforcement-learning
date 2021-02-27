import random
import numpy as np

# probably a better file name is single_neuron_output


def render_learnt_model(env, model, steps):
    random.seed(42)

    obs = env.reset()
    for _ in range(steps):
        env.render()
        left_prob = model.predict(obs[np.newaxis])
        action = random.uniform(0, 1) > left_prob
        action = int(action)
        obs, reward, done, info = env.step(action)
        if done:
            print("Game over")
            break
