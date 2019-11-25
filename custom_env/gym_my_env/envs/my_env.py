import os
import random

import tensorflow as tf
import gym
from gym import spaces
import numpy as np
import cv2

from custom_env.gym_my_env.envs.viewport import Viewport
from dataset import DataGenerator, Sal360
from GAIL.jupyter.utils import *


n_samples = 1
width = 224
height = 224
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
        self.observation = None
        self.mode = None
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
    def step(self, action=None, test=False):
        try:
            x_data, y_data = next(self.x_iter), next(self.y_iter)
            temp_y = [0] * 4
        except StopIteration:
            return None, None, True, None

        frame_idx = int(x_data[6] - x_data[5])
        frames = []
        ret = None

        for i in range(frame_idx):
            ret, frame = self.cap.read()
            if ret:
                if action is None:
                    w, h = x_data[2] * self.width, x_data[1] * self.height
                    self.view.set_center(np.array([w, h]))
                else:
                    self.view.move((action[0] * self.width, action[1] * self.height))
                frame = self.view.get_view(frame)
                try:
                    frame = cv2.resize(frame, (width, height))
                    # frame = standardizaation_frame(frame)
                    frames.append(frame)
                except Exception as e:
                    pass
            else:
                self.cap.release()
                return None, 0, not ret, None
        if len(frames) < 1:
            return None, 0, True, None
        else:
            self.observation = frames
            yy = find_direction(y_data)
            return self.observation, 0, not ret, ((x_data[2], x_data[1]), y_data)

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
        # print('start video: ', target_video)
        return target_video

    def render(self):
        for frame in self.observation:
            cv2.imshow("video", frame)
            # "q" --> stop and print info
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# test agent
if __name__ == '__main__':
    env = gym.make("my-env-v0")
    expert_ob, expert_ac, videos = generate_expert_trajectory(env, 10, render=True)
