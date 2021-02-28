import random
import numpy as np


def play_model(env, model, steps):
    random.seed(42)

    obs = env.reset()
    for _ in range(steps):
        env.render()
        probs = model.predict(obs[np.newaxis])
        action = np.argmax(probs)
        obs, reward, done, info = env.step(action)
        if done:
            print("Game over")
            break
