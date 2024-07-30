from gym.envs.registration import register
import gym
import numpy as np
import io

train_tasks = [((255,0,128), 'left'), ((255,0,128), 'right'), ((255,0,128), 'top'),
                ((128,255,0), 'left'), ((128,255,0), 'right'), ((128,255,0), 'top'),
                ((0,128,255), 'left'), ((0,128,255), 'right'), ((0,128,255), 'top')]

test_tasks = [((0,255,128), 'left'), ((0,255,128), 'right'), ((0,255,128), 'top'),
                ((255,128,0), 'left'), ((255,128,0), 'right'), ((255,128,0), 'top'),
                ((128,0,255), 'left'), ((128,0,255), 'right'), ((128,0,255), 'top')]


register(
     id="IllustrativeCMDPContinuous-v0",
     entry_point="illustrative_env:IllustrativeCMDPContinuous",
)

register(
     id="IllustrativeCMDPDiscrete-v0",
     entry_point="illustrative_env:IllustrativeCMDPDiscrete",
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
        # The agent is on the top
        return 3


env = gym.make('IllustrativeCMDPContinuous-v0', tasks=train_tasks)
env.seed(88)
obs = env.reset()

ep_obs = [obs]
ep_rewards = []
ep_dones = []
ep_actions = []
for i in range(6):
     action = optimal_policy(env)
     obs, reward, done, _ = env.step(action)
     ep_obs.append(obs)
     ep_rewards.append([reward])
     ep_dones.append(done)
     ep_actions.append([action])

     print(reward, done)
     if done:
            episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_actions), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones)}
            with io.BytesIO() as bs:
                np.savez_compressed(bs, **episode)
                bs.seek(0)
                with open(f'datasets/illustrative/episode_{i}.npy', "wb") as f:
                    f.write(bs.read())
            obs = env.reset()
            ep_obs = [obs]
            ep_rewards = []
            ep_dones = []
            ep_actions = []