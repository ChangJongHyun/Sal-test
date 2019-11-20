import matplotlib.pyplot as plt
import numpy as np
import cv2


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
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
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


def generate_expert_trajectory(env, num_envs, render=False):
    print("collect expert data...")
    obs = []
    acs = []
    target_videos = []
    for i in range(num_envs):
        target_video = env.reset()
        target_videos.append(target_video)
        while True:
            observation, r, done, action = env.step()
            if done:
                break
            else:
                if render:
                    env.render()
                obs.append(observation)
                acs.append(action)
    # expert_obs, expert_acs
    print("expert_obs size : ", np.shape(obs))
    print("expert_acs size : ", np.shape(acs))
    return obs, acs, target_videos
