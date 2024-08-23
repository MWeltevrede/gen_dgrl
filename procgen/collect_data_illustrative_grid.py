from gym.envs.registration import register
import gym
import numpy as np
import io, os

# C4 Rotations
train_tasks = [(45,-45,0), (45,-45,90), (45,-45,180), (45,-45,270)]

# # Random Rotations
# train_tasks = [(45,-45,-6), (45,-45,94), (45,-45,171), (45,-45,289)]

test_tasks = [(45,-45,-11), (45,-45,11), (45,-45,65), (45,-45,99), (45,-45,167), (45,-45,204), (45,-45,259), (45,-45,325)]

register(
     id="ControlIllustrativeCMDP-v0",
     entry_point="control_illustrative_env:ControlIllustrativeCMDP",
)

def optimal_policy(step):
    # a handcrafted (basically) optimal policy
    if step < 12:
        return [-2, 2]
    else:
        return [2, 2]


env = gym.make('ControlIllustrativeCMDP-v0', tasks=train_tasks)
env.seed(88)
obs = env.reset()

dataset_dirname = 'datasets/control_illustrative'
os.makedirs(dataset_dirname, exist_ok=True)

ep_obs = [obs]
ep_rewards = []
ep_dones = []
ep_actions = []
for i in range(len(train_tasks)):
    done = False
    step = 0
    while not done:
        action = optimal_policy(step)
        obs, reward, done, _ = env.step(np.array(action))
        ep_obs.append(obs)
        ep_rewards.append([reward])
        ep_dones.append(done)
        ep_actions.append(action)
        step += 1

        if done:
            print(ep_rewards, ep_dones, step)
            episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_actions), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones)}
            with io.BytesIO() as bs:
                np.savez_compressed(bs, **episode)
                bs.seek(0)
                with open(dataset_dirname + f'/episode_{i}.npy', "wb") as f:
                    f.write(bs.read())
            obs = env.reset()
            ep_obs = [obs]
            ep_rewards = []
            ep_dones = []
            ep_actions = []