from gym.envs.registration import register
import gym
import numpy as np
import io, os

# alternating subgroup
train_tasks = [((255,0,128), 'left'), ((255,0,128), 'right'), ((255,0,128), 'top'), ((255,0,128), 'bottom'),
                ((0,255,128), 'left'), ((0,255,128), 'right'), ((0,255,128), 'top'), ((0,255,128), 'bottom')]

test_tasks = [((128,255,0), 'left'), ((128,255,0), 'right'), ((128,255,0), 'top'), ((128,255,0), 'bottom'),
                ((255,128,0), 'left'), ((255,128,0), 'right'), ((255,128,0), 'top'), ((255,128,0), 'bottom'),
                ((128,0,255), 'left'), ((128,0,255), 'right'), ((128,0,255), 'top'), ((128,0,255), 'bottom'),
                ((0,128,255), 'left'), ((0,128,255), 'right'), ((0,128,255), 'top'), ((0,128,255), 'bottom')]

# # Random Colours
# train_tasks = [((206,16,220), 'left'), ((206,16,220), 'right'), ((206,16,220), 'top'), ((206,16,220), 'bottom'),
#                 ((86,185,105), 'left'), ((86,185,105), 'right'), ((86,185,105), 'top'), ((86,185,105), 'bottom')]

# test_tasks = [((237,48,217), 'left'), ((237,48,217), 'right'), ((237,48,217), 'top'), ((237,48,217), 'bottom'),
#                 ((52,128,100), 'left'), ((52,128,100), 'right'), ((52,128,100), 'top'), ((52,128,100), 'bottom'),
#                 ((35,21,88), 'left'), ((35,21,88), 'right'), ((35,21,88), 'top'), ((35,21,88), 'bottom'),
#                 ((213,109,113), 'left'), ((213,109,113), 'right'), ((213,109,113), 'top'), ((213,109,113), 'bottom')]

register(
     id="GridIllustrativeCMDPContinuous-v0",
     entry_point="grid_illustrative_env:IllustrativeCMDPContinuous",
)

register(
     id="GridIllustrativeCMDPDiscrete-v0",
     entry_point="grid_illustrative_env:IllustrativeCMDPDiscrete",
)

def optimal_policy(env):
    agent_location = env._agent_location
    if agent_location[0] < env.arm_length:
        # The agent is on the left
        return 0
    if agent_location[0] > env.arm_length:
        # The agent is on the right
        return 2
    if agent_location[1] < env.arm_length:
        # The agent is on the top
        return 1
    if agent_location[1] > env.arm_length:
        # The agent is on the bottom
        return 3


env = gym.make('GridIllustrativeCMDPContinuous-v0', tasks=train_tasks, arm_length=6)
env.seed(88)
obs = env.reset()

dataset_dirname = 'datasets/grid_illustrative'
os.makedirs(dataset_dirname, exist_ok=True)

ep_obs = [obs]
ep_rewards = []
ep_dones = []
ep_actions = []
for i in range(len(train_tasks)):
    done = False
    step = 0
    while not done:
        action = optimal_policy(env)
        obs, reward, done, _ = env.step(np.array(action))
        ep_obs.append(obs)
        ep_rewards.append([reward])
        ep_dones.append(done)
        ep_actions.append([action])
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