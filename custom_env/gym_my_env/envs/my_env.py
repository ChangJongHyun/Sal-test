import os
import random
import time

import tensorflow as tf
import gym
from gym import spaces
import numpy as np
import cv2

from custom_env.gym_my_env.envs.viewport import Viewport
from dataset import Sal360
from DDPG.utils import get_SalMap_info, read_SalMap

n_samples = 1
width = 112
height = 112
n_channels = 3


class MyEnv(gym.Env):

    def __init__(self, test=False):
        self.test = test
        type = 'train'  # default is train
        self.train, self.validation, self.test = Sal360.load_sal360v2()
        self.width, self.height = 3840, 1920
        self.x_dict, self.y_dict = None, None
        self.x_iter, self.y_iter = None, None
        self.target_videos = None
        self.video_path = os.path.join('sample_videos', type, '3840x1920')
        self.files = os.listdir(self.video_path)
        self.action_space = spaces.Box(-1, 1, [2])  # -1,1 사이의 1x2 벡터
        self.observation_space = spaces.Box(low=0, high=255, shape=(width, height, n_channels))

        self.cap = None  # video input
        self.view = None  # viewport's area / current state
        self.saliency_view = None
        self.observation = None
        self.mode = None
        self.saliency_info = get_SalMap_info()

        self.set_dataset()

    def set_dataset(self, type='train'):
        self.video_path = os.path.join('sample_videos', type, '3840x1920')
        if type == 'train':
            self.x_dict, self.y_dict = self.train[0], self.train[1]
            self.files = os.listdir(os.path.join('sample_videos', 'train', '3840x1920'))
            self.target_videos = self.files
        elif type == 'validation':
            self.x_dict, self.y_dict = self.validation[0], self.validation[1]
            self.files = os.listdir(os.path.join('sample_videos', 'train', '3840x1920'))
            self.target_videos = self.files
        elif type == 'test':
            self.x_dict, self.y_dict = self.test[0], self.test[1]
            self.files = os.listdir(os.path.join('sample_videos', 'test', '3840x1920'))
            self.target_videos = self.files
        else:
            raise NotImplementedError
        self.mode = type

    # return -> observation, reward, done(bool), info
    def step(self, action=None):
        try:
            x_data, y_data = next(self.x_iter), next(self.y_iter)
        except StopIteration:
            return None, None, True, None
        lat, lng, start_frame, end_frame = x_data[2], x_data[1], int(x_data[5]), int(x_data[6])

        if action is None:
            self.view.set_center((lat, lng), normalize=True)
            self.saliency_view.set_center((lat, lng), normalize=True)
        else:
            self.view.move(action)
            self.saliency_view.move(action)

        self.observation = [cv2.resize(self.view.get_view(f), (width, height), interpolation=cv2.INTER_AREA) for f in
                            self.video[start_frame - 1:end_frame]]

        """saliency mode"""
        saliency_observation = [self.saliency_view.get_view(f) for f in self.saliency[start_frame - 1:end_frame]]

        total_sum = np.sum(self.saliency[start_frame - 1:end_frame])
        observation_sum = np.sum(saliency_observation)
        reward = observation_sum / total_sum
        """end saliency mode"""

        # if reward < 0.05:
        #     reward += -1
        # elif reward > 0.25:
        #     reward += 1

        if len(self.observation) != 6:
            self.observation = normalize(self.observation)
        return self.observation, reward, False, y_data

    # 나중에 여러개의 영상을 학습하려면 iterate하게 영상을 선택하도록 g바꿔야함.
    def reset(self, target_video=None, set_data='train'):
        self.set_dataset(set_data)
        if target_video is None:
            target_video = random.choice(self.target_videos)
        random_idx = random.randint(0, len(self.x_dict[target_video]) - 1)
        random_x, random_y = self.x_dict[target_video][random_idx], self.y_dict[target_video][random_idx]
        self.cap = cv2.VideoCapture(os.path.join(self.video_path, target_video))
        self.x_iter, self.y_iter = iter(random_x), iter(random_y)
        self.view = Viewport(self.width, self.height)
        self.video = []
        """saliency mode"""
        self.saliency_view = Viewport(self.saliency_info[target_video][1], self.saliency_info[target_video][2])
        try:
            self.saliency = read_SalMap(self.saliency_info[target_video])
        except ValueError:
            print("Cannot find saliency map " + target_video + "... reset environmnet!")
            return self.reset()
        """end saliency mode"""
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.video.append(frame)
            else:
                self.cap.release()
                break

        """initial state"""
        x_data, y_data = next(self.x_iter), next(self.y_iter)
        lat, lng, start_frame, end_frame = x_data[2], x_data[1], int(x_data[5]), int(x_data[6])
        self.view.set_center((lat, lng), normalize=True)
        self.observation = [cv2.resize(self.view.get_view(f), (width, height), interpolation=cv2.INTER_AREA) for f in
                            self.video[start_frame - 1:end_frame]]
        if len(self.observation) != 6:
            self.observation = normalize(self.observation)
        return self.observation, y_data, target_video

    def render(self):
        for f in self.observation:
            cv2.imshow('origin', f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def normalize(obs):
    if len(obs) < 6:
        for i in range(6 - len(obs)):
            obs = np.concatenate([obs, [obs[-1]]])
        return obs
    elif len(obs) > 6:
        obs = obs[:6]
        return obs
    else:
        return obs
