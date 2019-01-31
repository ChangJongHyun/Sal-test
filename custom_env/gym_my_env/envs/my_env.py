import gym
from gym import spaces

import numpy as np
import cv2
import os
from custom_env.gym_my_env.envs.viewport import Viewport
from PIL import Image

delta = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])



class MyEnv(gym.Env):

    def __init__(self):
        self.files = os.listdir("src")
        self.action_space = spaces.Box(0, 2, [2])  # 0~2 사이의 1x2 벡터
        width, height = self.cap.get(3), self.cap.get(4)
        # self.observation_space = spaces.Box(low=np.array([0, 0]),
        #                                     high=np.array([width, height]), dtype=int)

        self.observation_space = None
        self.result = None  # object detection info
        self.cap = None  # video input
        self.agent = None  # viewport
        self.view = None  # viewport's area / current state
        self.ret, self.frame = None, None  # done, current video frame
        self.done = False

    def print_info(self):
        print(self.result)

    def gen(self):
        return np.random.randint(0, len(self.files))

    def read_frame(self):
        return self.cap.read()  # return ret, frame

    # return -> observation, reward, done(bool), info
    def step(self, action):
        self.ret, self.frame = self.read_frame()
        assert self.action_space.contains(action)
        self.done = not self.ret
        if self.done is True:
            return None, None, self.done, None
        else:
            self.agent.move(delta[action])
            self.view = self.agent.get_view(frame=self.frame)
            self.view = cv2.resize(self.view, (84, 84))
            # TODO rewrd
            return self.view, reward, self.done, None

    # 나중에 여러개의 영상을 학습하려면 iterate하게 영상을 선택하도록 바꿔야함.
    def reset(self):
        file = self.files[self.gen()]
        self.cap = cv2.VideoCapture('src/' + file)
        self.agent = Viewport(self.cap.get(3), self.cap.get(4))
        width, height = self.cap.get(3), self.cap.get(4)
        self.observation_space = spaces.Box(low=np.array([0, 0]),
                                            high=np.array([width, height]), dtype=int)
        ret, frame = self.cap.read()
        self.view = self.agent.get_view(frame)
        self.view = cv2.resize(self.view, (84, 84))
        self.done = not ret
        print("target file: ", file)
        return self.view

    def render(self, mode='full'):
        if mode == 'viewport':
            cv2.imshow("viewport", self.view)
        elif mode == "full":
            rec_point = self.agent.get_rectangle_point()
            f = self.frame
            for rec in rec_point:
                cv2.rectangle(f, rec[0], rec[1], (0, 0, 255), thickness=2)
            cv2.imshow("video", f)

        # "q" --> stop and print info
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.print_info()
            cv2.waitKey(0)

        return f


# test agent
if __name__ == '__main__':
    a = os.listdir("src")
    print(a)
    env = gym.make("my-env-v0")
    env.reset()
    reward = 0
    episodes = 1
    while True:
        state, r, done, _ = env.step(3)
        print(np.shape(state))
        if done:
            print(episodes, "'s reward is ", reward)
            reward = 0
            env.reset()
        else:
            env.render()
            reward += r