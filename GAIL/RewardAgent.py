import os

import numpy as np
import tensorflow as tf
import gym

from GAIL.Rward import Reward

if __name__ == '__main__':
    env = gym.make('my-env-v0')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    model = Reward(sess, env)

    print("Start Learn")

    train_step = 50_000
    test_step = 1000

    minibatch_size = 20
    max_step_per_episode = 600

    global_step = 0
    episode_num = 0

    while global_step < train_step:

        episode_num += 1

        if episode_num >= 500:
            break

        # obs, target_video, action = env.reset()  # Reset environment
        target_video = env.reset()

        total_reward = 0
        predicted_total_reward = 0
        done = False

        obs, acs, rewards = [], [], []

        while not done and episode_num < max_step_per_episode and global_step < train_step:  ### KH: reset every 200 steps

            global_step += 1

            ob, reward, done, action = env.step()
            predict_reward = model.get_reward([ob], [action])
            if done:
                break
            else:
                obs.append(ob)
                acs.append(acs)
                rewards.append(reward)
                predicted_total_reward += predict_reward
                total_reward += reward

                if len(obs) == minibatch_size:
                    model.update(global_step, obs, acs, rewards)
                    obs, acs, rewards = [], [], []
        print("[ target video: {}, train_ep: {}, (true, logits): ({}, {})"
              .format(train_step, episode_num, total_reward, predicted_total_reward))
        if episode_num is not 0 and episode_num % 10 == 0:
            model.save(os.path.join('model', 'REWARD'), episode_num)