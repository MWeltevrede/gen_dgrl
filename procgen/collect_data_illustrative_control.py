from gym.envs.registration import register
import gym
import numpy as np
import io, os
import torch
import copy
import json

from stable_baselines3 import SAC

def optimal_policy(step):
    # a handcrafted (basically) optimal policy
    if step < 12:
        return [-2, 2]
    else:
        return [2, 2]

def generate_dataset(tasks, dataset_name, model):
    env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks)
    env.seed(88)
    obs = env.reset()

    dataset_dirname = f'datasets/{dataset_name}'
    os.makedirs(dataset_dirname, exist_ok=True)

    ep_obs = [obs]
    ep_rewards = []
    ep_dones = []
    ep_actions = []
    total_steps = 0
    for i in range(len(tasks)):
        done = False
        step = 0
        while not done:
            obs = torch.from_numpy(obs).float().to('cuda')
            if len(obs.shape) == 1:
                # add batch dimension
                obs = obs.unsqueeze(0)
            with torch.no_grad():
                action = model.policy._predict(obs, deterministic=True)
            # using numpy, if action is of shape [1,1] then convert it to [1]
            action = action.cpu().numpy()
            if len(action.shape) == 2:
                action = action[0]

            # action = optimal_policy(step)
            obs, reward, done, _ = env.step(np.array(action))
            ep_obs.append(obs)
            ep_rewards.append([reward])
            ep_dones.append(done)
            ep_actions.append(action)
            step += 1

            if done:
                total_steps += step
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
        i += 1

    return total_steps

# #### Fixed Pose
# # SO(2) subgroups
# c2_train_tasks = [(45,-45,0), (45,-45,180)]
# c4_train_tasks = [(45,-45,0), (45,-45,90), (45,-45,180), (45,-45,270)]
# c8_train_tasks = [(45,-45,0), (45,-45,45), (45,-45,90), (45,-45,135), (45,-45,180), (45,-45,225), (45,-45,270), (45,-45,315)]

# register(
#      id="ControlIllustrativeCMDP-v0",
#      entry_point="control_illustrative_env:ControlIllustrativeCMDP",
# )

# model = SAC.load("sac_control_illustrative_full")

# sizec2 = generate_dataset(c2_train_tasks, f'control_illustrative_base_c2', model)
# sizec4 = generate_dataset(c4_train_tasks, f'control_illustrative_base_c4', model)
# sizec8 = generate_dataset(c8_train_tasks, f'control_illustrative_base_c8', model)

# with open('datasets/dataset_sizes_bases.txt', 'w') as file:
#     print(f"base_c2: {sizec2}", file=file)
#     print(f"base_c4: {sizec4}", file=file)
#     print(f"base_c8: {sizec8}", file=file)






### Randomly generated
for set_id in range(20):
    # set_id = 1
    np.random.seed(set_id)

    num_base_tasks = 4
    base_tasks = np.array([[np.random.randint(0, 360),np.random.randint(0, 360),0], [np.random.randint(0, 360),np.random.randint(0, 360),90], [np.random.randint(0, 360),np.random.randint(0, 360),180], [np.random.randint(0, 360),np.random.randint(0, 360),270]], dtype=np.int64)
    print(base_tasks.shape)

    base_da_tasks = []
    for i in range(num_base_tasks):
        a = copy.deepcopy(base_tasks)
        a[:, -1] = np.roll(base_tasks[:, -1], i+1)
        base_da_tasks.append(a)
    base_da_tasks = np.concatenate(base_da_tasks, axis=0)

    base_random_tasks = []
    for i in range(num_base_tasks):
        a = np.random.randint(0, 360, (num_base_tasks, 3))
        a[:, -1] = copy.deepcopy(base_tasks[:, -1])
        base_random_tasks.append(a)
    base_random_tasks = np.concatenate(base_random_tasks, axis=0)

    # unreachablex8_tasks = [copy.deepcopy(base_tasks)]
    # for i in range(num_base_tasks-1):
    #     a = np.random.randint(0, 360, (num_base_tasks, 3))
    #     unreachablex8_tasks.append(a)
    # unreachablex8_tasks = np.concatenate(unreachablex8_tasks, axis=0)

    # unreachablex4_tasks = [copy.deepcopy(base_tasks)]
    # for i in range(3):
    #     a = np.random.randint(0, 360, (num_base_tasks, 3))
    #     unreachablex4_tasks.append(a)
    # unreachablex4_tasks = np.concatenate(unreachablex4_tasks, axis=0)

    # unreachablex2_tasks = [copy.deepcopy(base_tasks)]
    # a = np.random.randint(0, 360, (num_base_tasks, 3))
    # unreachablex2_tasks.append(a)
    # unreachablex2_tasks = np.concatenate(unreachablex2_tasks, axis=0)


    register(
        id="ControlIllustrativeCMDP-v0",
        entry_point="control_illustrative_env:ControlIllustrativeCMDP",
    )
        
    model = SAC.load("sac_control_illustrative_full")

    base_size = generate_dataset(base_tasks, f'control_illustrative_base_{set_id}', model)
    da_size = generate_dataset(base_da_tasks, f'control_illustrative_base_da_{set_id}', model)
    random_size = generate_dataset(base_random_tasks, f'control_illustrative_base_random_{set_id}', model)
    with open(f'datasets/dataset_sizes_random_poses_{set_id}.txt', 'w') as file:
        print(f"base: {base_size}", file=file)
        print(f"base + da: {da_size}", file=file)
        print(f"base + random: {random_size}", file=file)

    # save tasks
    with open(f"datasets/task_sets_{set_id}.json", 'w') as file:
        json.dump({'base':base_tasks.tolist(), 'base + da':base_da_tasks.tolist(), 'base + random':base_random_tasks.tolist()}, file)
