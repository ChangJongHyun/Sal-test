import matplotlib.pyplot as plt
import numpy as np


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
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


def test_env(model, env, vis=False):
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if vis:
            env.render()
        ac = model.get_action([ob])[0]
        next_ob, reward, done, _ = env.step(ac)
        ob = next_ob
        total_reward += reward
    return total_reward


def generate_expert_trajectory(env, num_envs):
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
                obs.append(observation)
                acs.append(action)
    # expert_obs, expert_acs
    print("expert_obs size : ", np.shape(obs))
    print("expert_acs size : ", np.shape(acs))
    return obs, acs, target_videos
