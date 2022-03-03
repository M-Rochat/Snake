from stable_baselines3 import PPO
import os
import time
from snake_env import SnakeEnv

time_sig = str(int(time.time()))[4:]
models_dir = f"models/PPO-{time_sig}"
log_dir = f"logs/PPO-{time_sig}"


def train_model(model):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    TIMESTEPS = 10000
    for i in range(1, 100):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=(i == 1), tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS * i}")


def play_model(model):
    obs = env.reset()
    i = 0
    while i < 3:
        action, _state = model.predict(observation=obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        env.render(done=done)
        if done:
            i += 1
            obs = env.reset()


env = SnakeEnv()
env.reset()
# model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log=logdir)
model = PPO.load(path="trained_model.zip", env=env, verbose=1, tensorboard_log=log_dir)
# train_model(model)
play_model(model)
env.close()
