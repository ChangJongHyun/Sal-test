import os
import csv
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import math
import cv2
import time
import keras

from custom_env.gym_my_env.envs.viewport import Viewport

fps_list = {'01_PortoRiverside.mp4': 25, '02_Diner.mp4': 30, '03_PlanEnergyBioLab.mp4': 25,
            '04_Ocean.mp4': 30, '05_Waterpark.mp4': 30, '06_DroneFlight.mp4': 25, '07_GazaFishermen.mp4': 25,
            '08_Sofa.mp4': 24, '09_MattSwift.mp4': 30, '10_Cows.mp4': 24, '11_Abbottsford.mp4': 30,
            '12_TeatroRegioTorino.mp4': 30, '13_Fountain.mp4': 30, '14_Warship.mp4': 25, '15_Cockpit.mp4': 25,
            '16_Turtle.mp4': 30, '17_UnderwaterPark.mp4': 30, '18_Bar.mp4': 25, '19_Touvet.mp4': 30}


class Sal360:
    def __init__(self):
        # self.data, self.action = self.load_sal360_dataset()
        # self.input_data = self.load_video_dataset()
        pass

    def load_sal360_dataset(self):
        path = os.path.join("dataset", "H", "Scanpaths")
        data = []
        actions = []

        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file)) as f:
                row = csv.reader(f)
                data.append([list(map(lambda x: [float(i) for i in x], list(row)[1:]))])

        np_data = np.array(data).reshape((-1, 7))

        for idx, i in enumerate(np_data):
            if 99 == int(np_data[idx][0]):
                action = (0, 0)
            else:
                action = (np_data[idx + 1][1] - np_data[idx][1], np_data[idx + 1][2] - np_data[idx][1])
            actions.append(action)

        actions = np.array(actions)

        np_data = np_data.reshape((19, 57, 100, 7))
        actions = actions.reshape((19, 57, 100, 2))

        _x_train, x_test = np_data[:15], np_data[15:]
        _y_train, y_test = actions[:15], actions[15:]

        x_train, x_validation = np.array([i[:45] for i in _x_train]), np.array([i[45:] for i in _x_train])
        y_train, y_validation = np.array([i[:45] for i in _y_train]), np.array([i[45:] for i in _y_train])

        print("shape : (# of video, # of person, # of data per video, # of data)")
        print("shape of train set x, y : ", x_train.shape, y_train.shape)
        print("shape of validation set x, y : ", x_validation.shape, y_validation.shape)
        print("shape of test set x, y : ", x_test.shape, y_test.shape)

        return (x_train, y_train), (x_validation, x_validation), (x_test, y_test)

    def plot_state_data(self, data):
        assert np.shape(data) == 25, 2
        data = np.transpose(data)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.plot(data[0], data[1], range(100))
        plt.title("test")
        plt.show()

    def kl_divergence(self):

        state = np.array([(i[1] * 3840, i[2] * 1920) for i in self.data]).reshape([19, 57, 100, 2])
        state_prob = np.array([(i[1], i[2]) for i in self.data]).reshape([19, 57, 100, 2])

        base = state[0][0]  # 100, 2
        base_T = state[0][0].transpose()  # 2, 100
        base_prob = state_prob[0][0]
        base_prob_T = state_prob[0][0].transpose()

        def l2norm(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        for i, ii in zip(state_prob[0], state[0]):  # 57 2 100  # KLdivergence, L2norm
            i_T = i.transpose()
            distance = [l2norm(p1, p2) for p1, p2 in zip(base, ii)]

            kld_x = stats.entropy(pk=base_prob_T[0], qk=i_T[0])
            kld_y = stats.entropy(pk=base_prob_T[1], qk=i_T[1])

            print("KL divergence (x, y) : ({}, {})".format(kld_x, kld_y))
            print("avg L2 norm : {}".format(min(distance)))


def data_generator():
    train, validation, test = Sal360().load_sal360_dataset()
    width = 3840
    height = 1920
    view = Viewport(width, height)

    while True:
        for video, _x, _y in zip(sorted(os.listdir(os.path.join('sample_videos', 'train', '3840x1920'))),
                                 train[0], train[1]):  # tr: 45, 100, 7
            trace_length = fps_list[video]
            for x, y in zip(_x, _y):  # _x : 100, 7 _y : 100, 2
                cap = cv2.VideoCapture(os.path.join('sample_videos/train/3840x1920/', video))
                x_iter = iter(x)
                y_iter = iter(y)
                x_data = next(x_iter)
                y_data = next(y_iter)
                frame_idx = x_data[6] - x_data[5] + 1
                index = 0

                sequenceX = []
                sequenceY = []

                while True:
                    ret, frame = cap.read()
                    if ret:
                        index += 1
                        if index == frame_idx:
                            w, h = x_data[1] * width, x_data[2] * height
                            view.set_center(np.array([w, h]))
                            frame = view.get_view(frame)
                            frame = cv2.resize(frame, (224, 224))

                            x_data = next(x_iter)
                            y_data = next(y_iter)
                            y_data = [y_data[0] * width, y_data[1] * height]
                            sequenceX.append(frame)
                            sequenceY.append(y_data)

                            frame_idx = x_data[6] - x_data[5] + 1
                            index = 0
                            if len(sequenceX) == trace_length:
                                print("shape x : ", np.shape(sequenceX))
                                print("shape y : ", np.shape(sequenceY))
                                yield sequenceX, sequenceY
                                sequenceX = []
                                sequenceY = []
                    else:
                        if len(sequenceX) > 0 and len(sequenceX) != trace_length:
                            for i in range(trace_length - len(sequenceX)):
                                sequenceX.append(sequenceX[-1])
                                sequenceY.append(sequenceY[-1])
                            print("shape x : ", np.shape(sequenceX))
                            print("shape y : ", np.shape(sequenceY))
                            yield sequenceX, sequenceY
                        break

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    gen = data_generator()
