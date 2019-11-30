import os
import re
from collections import deque
import random

from gym.spaces import Discrete, Box
import matplotlib.pyplot as plt
import numpy as np
import cv2


SalMapDir = os.path.join('dataset', 'H', 'SalMaps')
dtypes = {16: np.float16,
          32: np.float32,
          64: np.float64}


def normalize_frame(frame_in):
    min_value = np.min(frame_in)
    max_value = np.max(frame_in)
    return (np.array(frame_in) - min_value) / (max_value - min_value)


def standardizaation_frame(frame_in):
    mean_value = np.mean(frame_in)
    std_value = np.std(frame_in)
    return (np.array(frame_in) - mean_value) / std_value


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95, scalar=True):
    gae = 0
    returns = []
    if scalar:
        delta = rewards + gamma * next_value * masks - values
        gae = delta + gamma * tau * masks * gae
        return gae + values
    else:
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks - values[step]
            gae = delta + gamma * tau * masks * gae
            returns.insert(0, gae + values[step])
        return returns


def ppo_iter(mini_batch_size, obs, acs, returns, advantage):
    batch_size = obs.shape[0]  # observation 의 수
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)  # mini_batch_size 만큼 sampling
        yield (obs[rand_ids, :], acs[rand_ids, :],
               returns[rand_ids, :], advantage[rand_ids, :])


def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


def test_env(model, env, render=False):
    target_video = env.reset()
    print("test env : ", target_video)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # out = cv2.VideoWriter('output_' + target_video + '.avi', fourcc, 25, 224, 224)
    total_reward = 0
    ob, _, _, action = env.step()  # initial state
    # for i in ob:
    #     out.write(i)
    while True:
        if render:
            env.render()
        ac = model.get_action([ob])[0]
        ac = np.resize(ac, [2])
        print('(real action, predicted action) : ({}, {})'.format(action, ac))
        next_ob, reward, done, action = env.step()
        # for i in next_ob:
        #     out.write(i)
        if done:
            break
        else:
            ob = next_ob
            total_reward += reward
    # out.release()
    return total_reward, target_video


def test_sl(model, env, set_data='test', render=False):
    target_video = env.reset(set_data=set_data)
    print("{} env : {}".format(set_data, target_video))
    real = []
    predicted = []
    while True:
        if render:
            env.render()
        next_ob, reward, done, action = env.step()
        if len(next_ob) == 4 and len(next_ob) >= 2:
            for i in range(5 - len(next_ob)):
                next_ob.append(next_ob[-1])
        elif len(next_ob) > 5:
            next_ob = next_ob[:5]
        ac = model.get_action([next_ob])
        print('(real action, predicted action) : ({}, {})'.format(action, ac))
        if done:
            break
        else:
            real.append(action)
            predicted.append(ac)


def generate_expert_trajectory(env, num_envs, render=False, target_video=None):
    print("collect expert data...")
    obs = []
    acs = []
    target_videos = []
    for i in range(num_envs):
        if target_video is None:
            target_video1 = env.reset()
        else:
            env.reset(target_video=target_video)
        target_videos.append(target_video1)
        ob_s = []
        ac_s = []
        while True:
            observation, r, done, action = env.step()
            if done:
                break
            else:
                if render:
                    env.render()
                ob_s.append(observation)
                ac_s.append(action)
        obs.append(ob_s)
        acs.append(ac_s)
    # expert_obs, expert_acs
    print("expert_obs size : ", np.shape(obs))
    print("expert_acs size : ", np.shape(acs))
    print("train: ", target_videos)
    return obs, acs, target_videos


def make_normailize_action(action_data, w_h=None):
    tmp = [0] * 4
    if action_data[0] >= 0:
        if w_h is not None:
            tmp[0] = action_data[0] * w_h[0]
        else:
            tmp[0] = action_data[0]
    else:
        if w_h is not None:
            tmp[1] = -action_data[0] * w_h[0]
        else:
            tmp[1] = -action_data[0]

    if action_data[1] >= 0:
        if w_h is not None:
            tmp[2] = action_data[1] * w_h[1]
        else:
            tmp[2] = action_data[1]
    else:
        if w_h is not None:
            tmp[3] = -action_data[1] * w_h[1]
        else:
            tmp[3] = -action_data[1]
    return tmp


def find_direction(action_data):
    # ++, +-, -+, --, step
    direction_vector = [np.sign(action_data[0]), np.sign(action_data[1])]
    action_data = [abs(i) > 0.1 for i in action_data]
    return [i * j for i, j in zip(direction_vector, action_data)]


def get_SalMap_info():
    data = os.listdir(SalMapDir)
    data = [file for file in data if file.endswith(".bin")]
    print(data)
    get_file_info = re.compile("(\d+_\w+)_(\d+)x(\d+)x(\d+)_(\d+)b")
    # 각 영상의 name, width, height, dtype
    sal_map_info = dict()
    for i in data:
        info = get_file_info.findall(i)[0]
        sal_map_info[info[0] + '.mp4'] = [i] + list(map(lambda x: int(x), info[1:]))
    print(sal_map_info)
    return sal_map_info


def read_SalMap(info):
    full = []
    with open(os.path.join(SalMapDir, info[0]), "rb") as f:
        # Position read pointer right before target frame
        width, height, Nframe, dtype = info[1], info[2], info[3], info[4]
        for i in range(Nframe):
            f.seek(width * height * (i) * (dtype // 8))

            #  Read from file the content of one frame
            data = np.fromfile(f, count=width * height, dtype=dtypes[dtype])
            # Reshape flattened data to 2D image
            full.append(data.reshape([height, width]))
    return full


def observation_dim(ob_space):
    if isinstance(ob_space, Discrete):
        return list(ob_space.n)

    elif isinstance(ob_space, Box):
        return list(ob_space.shape)

    else:
        raise NotImplementedError


def action_dim(a_space):  ### KH: for continuous action task

    if isinstance(a_space, Discrete):
        return list(a_space.n)

    elif isinstance(a_space, Box):
        print(a_space.shape)
        return list(a_space.shape)

    else:
        raise NotImplementedError


class ReplayBuffer:
    def __init__(self, buffer_size=5000, minibatch_size=64):
        self.replay_memory_capacity = buffer_size  # capacity of experience replay memory
        self.minibatch_size = minibatch_size  # size of minibatch from experience replay memory for updates
        self.replay_memory = deque(maxlen=self.replay_memory_capacity)

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    def sample_from_memory(self):
        return random.sample(self.replay_memory, self.minibatch_size)

    def erase(self):
        self.replay_memory.popleft()

    def __len__(self):
        return len(self.replay_memory)


if __name__ == '__main__':
    pass
    # envs = gym.make("my-env-v0")
    # generate_expert_trajectory(envs, 3)
    # sal_info = get_SalMap_info()
    # vv = Viewport(2048, 1024)
    # for k, v in sal_info.items():
    #     data = read_SalMap(v)
    #     for i in data:
    #         total_sum = np.sum(i)
    #         if np.isnan(total_sum):
    #             np.nan_to_num(total_sum)
    #         i = vv.get_view(i)
    #         viewing_sum = np.sum(i)
    #         print(np.shape(i))
    #         cv2.imshow('test', i)
    #         print(viewing_sum / total_sum)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     print(np.shape(data))
