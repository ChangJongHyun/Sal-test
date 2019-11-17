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

    def __init__(self):
        self.type = 'train'
        self.train, self.validation, self.test = Sal360.load_sal360v2()
        self.x_dict, self.y_dict = None, None
        self.x_iter, self.y_iter = None, None
        self.target_videos = None
        self.DataGenerator = DataGenerator.generator_for_batch(width, height)
        # TODO 타겟 폴더변경
        # TODO EXPERT mode and AGENT mode ??
        self.files = os.listdir(os.path.join(video_path, self.type))
        self.action_space = spaces.Box(-1, 1, [2])  # 0~2 사이의 1x2 벡터
        self.observation_space = spaces.Box(low=0, high=255, shape=(n_samples, width, height, n_channels))

        self.cap = None  # video input
        self.agent = None  # viewport
        self.view = None  # viewport's area / current state
        self.ret, self.frame = None, None  # done, current video frame
        self.done = False
        self.observation = None

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

    def print_info(self):
        print(self.result)

    def gen(self):
        return np.random.randint(0, len(self.files))

    def read_frame(self):
        return self.cap.read()  # return ret, frame

    # return -> observation, reward, done(bool), info
    def step(self, action):
        # TODO 1개의 데이터셋의 값들을 받고 state, done 설정
        x_data, y_data = self.x_iter.next(), self.y_iter.next()
        frame_idx = x_data[6] - x_data[5] + 1
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
                return None, 0, not ret, None
        self.observation = np.array([frames])
        return self.observation, 0, not ret, None

    # 나중에 여러개의 영상을 학습하려면 iterate하게 영상을 선택하도록 g바꿔야함.
    def reset(self):
        # TODO 다른 영상의 다른 saliency로 변경
        self.cap.release()
        target_video = random.choice(self.target_videos)
        random_idx = random.randint(0, len(self.x_dict[target_video]))
        random_x, random_y = self.x_dict[target_video][random_idx], self.y_dict[target_video][random_idx]
        self.cap = cv2.VideoCapture(target_video)
        self.x_iter, self.y_iter = iter(random_x), iter(random_y)
        self.view = Viewport(3840, 1920)

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
    env.reset()
    while True:
        state, r, done, _ = env.step()
        print(np.shape(state))
        if done:
            env.reset()
        else:
            env.render()
