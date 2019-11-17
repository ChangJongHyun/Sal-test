import os
import random

import gym
from gym import spaces
import numpy as np
import cv2

from custom_env.gym_my_env.envs.viewport import Viewport
from dataset import DataGenerator, Sal360

delta = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])

n_samples = 1
width = 224
height = 224
n_channels = 3

video_path = os.path.join("sample_videos", "3840x1920")


class MyEnv(gym.Env):

    def __init__(self, mode='expert'):
        type = 'train'  # default is train
        self.mode = mode
        self.train, self.validation, self.test = Sal360.load_sal360v2()
        self.x_dict, self.y_dict = None, None
        self.x_iter, self.y_iter = None, None
        self.target_videos = None

        # TODO 타겟 폴더변경
        # TODO EXPERT mode and AGENT mode ??
        self.video_path = os.path.join('sample_videos', type, '3840x1920')
        self.files = os.listdir(self.video_path)
        self.action_space = spaces.Box(-1, 1, [2])  # 0~2 사이의 1x2 벡터
        self.observation_space = spaces.Box(low=0, high=255, shape=(width, height, n_channels))

        self.cap = None  # video input
        self.view = None  # viewport's area / current state
        self.observation = None

        self.set_dataset()

    def set_dataset(self, type='train'):
        if type == 'train':
            self.x_dict, self.y_dict = self.train[0], self.train[1]
            self.target_videos = self.files[:14]
        elif type == 'validation':
            self.x_dict, self.y_dict = self.validation[0], self.validation[1]
            self.target_videos = self.files[:14]
        elif type == 'test':
            self.x_dict, self.y_dict = self.test[0], self.test[1]
            self.target_videos = self.files[14:]
        else:
            raise NotImplementedError
        self.video_path = os.path.join('sample_videos', type, '3840x1920')

    def gen(self):
        return np.random.randint(0, len(self.files))

    def read_frame(self):
        return self.cap.read()  # return ret, frame

    # return -> observation, reward, done(bool), info
    def step(self, action=None):
        if self.mode == 'expert':
            pass
        elif self.mode == 'agent':
            pass
        else:
            raise NotImplemented
        # TODO 1개의 데이터셋의 값들을 받고 state, done 설정
        x_data, y_data = next(self.x_iter), next(self.y_iter)
        frame_idx = int(x_data[6] - x_data[5] + 1)
        frames = []
        ret = None
        for i in range(frame_idx):
            ret, frame = self.cap.read()
            if ret:
                w, h = x_data[1] * 3840, x_data[2] * 1920
                self.view.set_center(np.array([w, h]))
                frame = self.view.get_view(frame)
                frame = cv2.resize(frame, (width, height))
                frames.append(frame)
            else:
                self.cap.release()
                return None, 0, not ret, None

        self.observation = np.array([frames])
        return self.observation, 0, not ret, y_data

    # 나중에 여러개의 영상을 학습하려면 iterate하게 영상을 선택하도록 g바꿔야함.
    def reset(self, target_video=None):
        if target_video is None:
            target_video = random.choice(self.target_videos)
        # TODO 다른 영상의 다른 saliency로 변경
        random_idx = random.randint(0, len(self.x_dict[target_video]) - 1)
        random_x, random_y = self.x_dict[target_video][random_idx], self.y_dict[target_video][random_idx]
        self.cap = cv2.VideoCapture(os.path.join(self.video_path, target_video))
        self.x_iter, self.y_iter = iter(random_x), iter(random_y)
        self.view = Viewport(3840, 1920)
        print('start video: ', target_video)

    def render(self):
        for frame in self.observation[0]:
            cv2.imshow("video", frame)
            # "q" --> stop and print info
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


# test agent
if __name__ == '__main__':
    env = gym.make("my-env-v0")
    print(env.action_space.shape)
    print(env.observation_space.shape)
    env.reset()
    while True:
        state, r, done, ac = env.step()
        print(np.shape(state), ac, done)
        if done:
            env.reset()
