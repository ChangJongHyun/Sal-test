import os
import time

import numpy as np
import tensorflow as tf
import gym

from GAIL.jupyter.DDPG import DDPG
from GAIL.jupyter.Discriminator import Discriminator
from GAIL.jupyter.PPO import PPO
import matplotlib.pyplot as plt

from GAIL.jupyter.supervised import *
from custom_env.gym_my_env.envs import my_env
from GAIL.jupyter.utils import *

epochs = 10
num_envs = 16

ppo_hidden_size = 256
discriminator_hidden_size = 128
lr = 0.005  # 원래 2e-4
num_steps = 20
mini_batch_size = 5
ppo_epochs = 4
threshold_reward = -200

max_frames = 100000

test_rewards = []

envs = gym.make("my-env-v0")

early_stop = False
render = False


def supervised_learning_keras(model):
    target_video = None
    for epoch in range(500):
        if target_video is None:
            expert_ob, expert_ac, target_videos = generate_expert_trajectory(envs, 5)  # (num_envs * 100,)
        else:
            expert_ob, expert_ac, target_videos = generate_expert_trajectory(envs, 5,
                                                                             target_video=target_video)  # (num_envs * 100,)
        obs, acs = [], []
        for ob, ac in zip(expert_ob, expert_ac):
            print(np.shape(ob), np.shape(ac))
            if len(ob) == 4 and len(ob) >= 2:
                for i in range(5 - len(ob)):
                    ob.append(ob[-1])
            elif len(ob) > 5:
                ob = ob[:5]
            if len(ob) == 5:
                obs.append(ob)
                acs.append(acs)
            if len(obs) == 10:
                print(np.shape([obs], np.shape([acs])))
                print('train!')
                model.train_batch([obs], [acs])
                obs, acs = [], []
        if epoch % 5 == 0:
            model.save()
            model.plot()


def supervisd_learning(sess, save_dir, m, s, w, ckpt=None):
    frame_idx = 0
    step = 1
    if ckpt is not None:
        s.restore(sess, ckpt)
    target_video = None
    for epoch in range(500):
        if target_video is None:
            expert_ob, expert_ac, target_video = generate_expert_trajectory(envs, 5)  # (num_envs * 100,)
        else:
            expert_ob, expert_ac, target_video = generate_expert_trajectory(envs, 5,
                                                                             target_video=target_video)  # (num_envs * 100,)
        obs, acs = None, None
        for ob, ac in zip(expert_ob, expert_ac):
            if obs is None:
                obs = ob
                acs = ac[1]
            else:
                obs = np.concatenate([obs, ob])
                acs = np.sum([ac[1], acs], axis=1)
            if frame_idx is not 0 and frame_idx % 5 == 0:
                print('train! ', np.shape(obs), acs)
                obs = standardizaation_frame(obs)
                m.train(w, frame_idx, [obs], [acs])
                m.write_summary(w, step, [obs], [acs])
                obs, acs = None, None
                frame_idx = 0
                continue
            frame_idx += 1
            step += 1
        s.save(sess, save_dir + str(epoch) + '_sl.ckpt')


def gail_learn(sess, save_dir, ppo, discriminator, algo="PPO"):
    i_update = 0
    frame_idx = 0
    for epoch in range(500):
        print("epoch # of ", epoch)
        expert_ob, expert_ac, target_videos = generate_expert_trajectory(envs, 1)  # (num_envs * 100,)
        print(" finish collect expert data!")
        observation_idx = 0
        obs, acs, rewards, values = [], [], [], []
        # same as expert trajectory video
        for ob, ac in zip(expert_ob, expert_ac):
            if len(ob) == 4 and len(ob) >= 2:
                for i in range(5 - len(ob)):
                    ob.append(ob[-1])
            elif len(ob) > 5:
                ob = ob[:5]

            action, value = ppo.get_action([ob]), ppo.get_value([ob])
            action = np.resize(action, [2])
            reward = discriminator.get_reward([ob], [action])
            value = value[0]
            reward = reward[0]
            obs.append(ob)
            acs.append(ac)
            rewards.append(reward)
            values.append(value)
            if i_update % 5 == 0:
                print(reward, ac, action)
                next_value = ppo.get_value(obs)
                next_value = next_value[0][0]
                returns = compute_gae(next_value, rewards, 1, values, scalar=False)
                advantages = returns - value
                ppo.assign_old_pi()
                print(np.shape(advantages), np.shape(returns))
                ppo.update(writer, frame_idx, obs, acs, returns, advantages)
                obs, acs, rewards, values = [], [], [], []
            discriminator.learn(writer, frame_idx, [ob], [action],
                                [ob], [ac])

            frame_idx += 1
            i_update += 1
            observation_idx += 1

        if epoch is not 0 and epoch % 10 == 0:
            saver.save(sess, os.path.join('model', save_dir + str(epoch) + '_.ckpt'))


def train_reward(sess, save_dir, w, s, m, ckpt=None):
    frame_idx = 0
    step_index = 0
    sal_map_info = get_SalMap_info()
    sal_viewport = Viewport(2048, 1024)
    if ckpt is not None:
        s.restore(sess, ckpt)
    for epoch in range(500):
        expert_ob, expert_ac, target_video = generate_expert_trajectory(envs, 5)  # (num_envs * 100,)

        for obs, acs, video in zip(expert_ob, expert_ac, target_video):
            video_index = 0
            try:
                info = sal_map_info[video]
                saliency_map = read_SalMap(info)
            except ValueError:
                continue
            concat_ob, concat_ac = None, None
            for ob, ac in zip(obs, acs):
                center = np.array([ac[0][0] * 2048, ac[0][1] * 1024])
                sal_viewport.set_center(center)

                if concat_ob is None:
                    concat_ob = ob
                    concat_ac = np.array(ac[1])
                else:
                    concat_ob = np.concatenate([concat_ob, ob])
                    concat_ac = concat_ac + np.array(ac[1])

                if step_index == 4:
                    sal_frame = []
                    if info[3] <= video_index + len(concat_ob):
                        frames = saliency_map[video_index:]
                    else:
                        frames = saliency_map[video_index:video_index + len(concat_ob)]
                    for f in frames:
                        f = sal_viewport.get_view(f)
                        sal_frame.append(f)
                    total_sum = np.sum([np.nan_to_num(frame) for frame in frames])
                    partition_sum = np.sum([np.nan_to_num(frame) for frame in sal_frame])
                    video_index += len(ob)
                    reward = np.nan_to_num([partition_sum / total_sum])
                    print('train! ', np.shape(concat_ob), np.shape(concat_ac), reward, video_index)
                    m.train(w, frame_idx, [ob], [ac[1]], [reward])
                    step_index = 0
                    video_index += len(concat_ob)
                    concat_ob, concat_ac = None, None
                    continue
                step_index += 1
                frame_idx += 1
        s.save(sess, save_dir + str(epoch) + '_sl.ckpt')

def test(sess, ckpt_path, model):
    saver = tf.train.Saver()

    saver.restore(sess, ckpt_path)

    # test_env(, envs)


if __name__ == '__main__':
    ob_shape = [None] + list(envs.observation_space.shape)  # n_samples + (width, height, channels)
    ac_shape = list(envs.action_space.shape)
    print(ac_shape)
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(
                                                                        memory_limit=1024 * 8)])
        except RuntimeError as e:
            print(e)

    # model = SL_model_keras(ob_shape, ac_shape, isRNN=True)
    # supervised_learning_keras(model)

    session = tf.InteractiveSession()
    # model = SL_model(session, ob_shape, ac_shape, 'sl', trainable=True, isRNN=True)
    # discriminator = Discriminator(session, ob_shape, ac_shape, name='D', isRNN=True)
    # ppo = PPO(session, ob_shape, ac_shape, lr, ppo_hidden_size, isRNN=True)
    reward_model = Reward(session, ob_shape, ac_shape, name='reward', isRNN=True)
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./log/', session.graph)
    ckpt_path = './model/model'
    save_path = './model/model_reward2_'
    # test(sess, ckpt_path)
    # gail_learn(session, save_path, ppo, discriminator)
    # supervisd_learning(session, save_path, model, saver, writer)
    # saver.restore(session, "model/model_ssl_3_sl.ckpt")
    # test_sl(model, envs)
    train_reward(session, save_path, writer, saver, reward_model)
