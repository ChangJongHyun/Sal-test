import numpy as np
import tensorflow as tf
import gym
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
frame_idx = 0
test_rewards = []

envs = gym.make("my-env-v0")

ob_shape = [None] + list(envs.observation_space.shape)  # n_samples + (width, height, channels)
ac_shape = list(envs.action_space.shape)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
ppo = PPO(sess, ob_shape, ac_shape, lr, ppo_hidden_size)
discriminator = Discriminator(sess, ob_shape, ac_shape, discriminator_hidden_size, lr, 'D')

sess.run(tf.global_variables_initializer())

i_update = 0
ob = envs.reset()
early_stop = False


for _ in range(epochs):
    expert_ob, expert_ac, target_videos = generate_expert_trajectory(envs, num_envs)

    while frame_idx < max_frames and not early_stop:
        i_update += 1
        # params
        for target in target_videos:
            values = []
            obs = []
            acs = []
            rewards = []
            masks = []
            entropy = 0

            envs.reset(target_video=target)
            ob, _, done, _ = envs.step()

            while True:
                ac, value = ppo.get_action(ob), ppo.get_value(ob)
                reward = discriminator.get_reward(ob, ac)
                next_ob, _, done, _ = envs.step(action=ac)
                if done:
                    break
                else:
                    values.append(value)
                    rewards.append(reward)
                    masks.append((1 - done))
                    obs.append(ob)
                    acs.append(ac)

                    frame_idx += 1

                    next_value = ppo.get_value(next_ob)
                    returns = compute_gae(next_value, rewards, masks, values)
                    advantages = np.array(returns) - np.array(values)
                    # Policy Update
                    if i_update % 3 == 0:
                        ppo.assign_old_pi()
                        ppo.update(obs, acs, returns, advantages)
                    print('expert obs : ', np.shape(expert_ob))
                    print('agent_ obs : ', np.shape(obs))

                    # Discriminator Update
                    discriminator.update(np.concatenate([expert_ob, obs]),
                                         np.concatenate([expert_ac, acs]))



# while frame_idx < max_frames and not early_stop:
#     i_update += 1
#
#     values = []
#     obs = []
#     acs = []
#     rewards = []
#     masks = []
#     entropy = 0
#
#     # 20 step 마다 discrim update 60 step 마다 policy update
#     for _ in range(num_steps):
#
#         ac = ppo.get_action(ob)
#         next_ob, _, done, action = envs.step(ac)
#         reward = discriminator.get_reward(ob, ac)
#
#         value = ppo.get_value(ob)
#         values.append(value)
#         # TODO [:, np.newaxis] 지워도 될거 같기도..
#         rewards.append(reward[:, np.newaxis])
#         masks.append((1 - done)[:, np.newaxis])
#
#         obs.append(ob)
#         acs.append(ac)
#
#         ob = next_ob
#         frame_idx += 1
#
#         if frame_idx % 1000 == 0:
#             test_reward = np.mean([test_env(ppo) for _ in range(10)])
#             test_rewards.append(test_reward)
#             plot(frame_idx, test_rewards)
#             if test_reward > threshold_reward: early_stop = True
#
#     next_value = ppo.get_value(next_ob)
#     returns = compute_gae(next_value, rewards, masks, values)
#
#     returns = np.concatenate(returns)
#     values = np.concatenate(values)
#     obs = np.concatenate(obs)
#     acs = np.concatenate(acs)
#     advantages = returns - values
#
#     # Policy Update
#     if i_update % 3 == 0:
#         ppo.assign_old_pi()
#         for _ in range(ppo_epochs):
#             for ob_batch, ac_batch, return_batch, adv_batch in ppo_iter(mini_batch_size, obs, acs, returns, advantages):
#                 ppo.update(ob_batch, ac_batch, return_batch, adv_batch)
#
#     # Discriminator Update
#     policy_ob_ac = np.concatenate([obs, acs], 1)
#     discriminator.update(np.concatenate([expert_ob_ac, policy_ob_ac], axis=0))
