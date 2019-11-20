import os

import numpy as np
import tensorflow as tf
import gym

from GAIL.jupyter.DDPG import DDPG
from GAIL.jupyter.Discriminator import Discriminator
from GAIL.jupyter.PPO import PPO
import matplotlib.pyplot as plt

from custom_env.gym_my_env.envs import my_env
from GAIL.jupyter.utils import *

epochs = 10
num_envs = 16

ppo_hidden_size = 256
discriminator_hidden_size = 128
lr = 3e-4
num_steps = 20
mini_batch_size = 5
ppo_epochs = 4
threshold_reward = -200

max_frames = 100000

test_rewards = []

envs = gym.make("my-env-v0")

ob_shape = [None] + list(envs.observation_space.shape)  # n_samples + (width, height, channels)
ac_shape = list(envs.action_space.shape)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

discriminator = Discriminator(sess, ob_shape, ac_shape, discriminator_hidden_size, lr, 'D')

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

writer = tf.summary.FileWriter('./log/', sess.graph)

early_stop = False
render = False


def learn(save_dir, algo="PPO"):
    if algo == "PPO":
        ppo = PPO(sess, ob_shape, ac_shape, lr, ppo_hidden_size)
    elif algo == "DDPG":
        ddpg = DDPG(sess, ob_shape, ac_shape, 1, -1)
    else:
        raise NotImplementedError

    i_update = 0
    frame_idx = 0
    for epoch in range(30):
        print("epoch # of ", epoch)
        expert_ob, expert_ac, target_videos = generate_expert_trajectory(envs, 5)  # (num_envs * 100,)
        print(" finish collect expert data!")
        print(" expert observation size ", np.shape(expert_ob))
        print(" expert action size ", np.shape(expert_ac))
        observation_idx = 0

        # same as expert trajectory video
        for target in target_videos:
            print('    train target : ', target)
            envs.reset(target_video=target)
            ob, _, done, _ = envs.step()  # initial state

            # iterate video
            while True:
                # ac, value = ppo.get_action([ob]), ppo.get_value([ob])
                ac = ddpg.choose_action([ob])
                ac = np.resize(ac, [2])
                reward = discriminator.get_reward([ob], [ac])
                # value = value[0]
                reward = reward[0]
                next_ob, _, done, _ = envs.step(action=ac)
                if done or len(expert_ob) <= observation_idx:
                    break
                else:
                    if render:
                        envs.render()

                    # Policy Update
                    if i_update % 3 == 0:
                        if algo == "PPO":
                            value = 0
                            next_value = ppo.get_value([next_ob])
                            next_value = next_value[0][0]
                            returns = compute_gae(next_value, reward[0], 1 - done, value[0])
                            advantages = returns - value

                            ppo.assign_old_pi()
                            ppo.update(writer, frame_idx, [ob], [ac], [[returns]], [advantages])
                        elif algo == "DDPG":
                            ddpg.update(writer, frame_idx, [ob], [ac], [reward], [next_ob], [done])
                        else:
                            raise NotImplementedError
                    # Discriminator Update
                    discriminator.update(writer, frame_idx, [expert_ob[observation_idx]], [expert_ac[observation_idx]],
                                         [ob], [ac])

                    frame_idx += 1
                    i_update += 1
                    observation_idx += 1
            print('    train finish target : ', target)
            # TODO write test code
            # test --> 영상저장
            # test_reward = np.mean([test_env(ppo, envs) for _ in range(10)])
            # print(epoch + " test reward : ", test_reward)
            # test_rewards.append(test_reward)
            # plot(frame_idx, test_rewards)
            # if test_reward > threshold_reward: early_stop = True
        if epoch is not 0 and epoch % 10 == 0:
            saver.save(sess, os.path.join('model', save_dir + str(epoch) + '_.ckpt'))


def test(sess, ckpt_path, model):
    saver = tf.train.Saver()

    saver.restore(sess, ckpt_path)

    # test_env(, envs)


if __name__ == '__main__':
    ckpt_path = './model/model/model_ddpg_.ckpt'
    save_path = './model/model_ddpg_'
    # test(sess, ckpt_path)
    learn(save_path, algo='DDPG')
