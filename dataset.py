import os
import csv
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import math
import cv2
import time


class VideoData:
    pass


class Sal360:
    def __init__(self):
        # self.data, self.action = self.load_sal360_dataset()
        # self.input_data = self.load_video_dataset()
        pass

    def load_video_dataset(self, resolution='high'):
        start = time.time()
        if resolution == 'high':
            path = os.path.join('sample_videos', '3840x1920')
        else:
            path = os.path.join('sample_videos', '320x160')
        video_data = []
        for video in os.listdir(path):
            cap = cv2.VideoCapture(os.path.join(path, video))
            while True:
                ret, frame = cap.read()

                if ret:
                    video_data.append(frame)
                    cv2.imshow('test', frame)
                else:
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        end = time.time()

        print("X 걸린시간: {}초".format(end - start))
        print(np.shape(video_data))
        return video_data

    def load_sal360_dataset(self):
        start = time.time()
        path = os.path.join("dataset", "H", "Scanpaths")
        data = []
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file)) as f:
                row = csv.reader(f)
                data.append([list(map(lambda x: [float(i) for i in x], list(row)[1:]))])
        result = []

        np_data = np.array(data).reshape((-1, 7))
        actions = []
        frames = [i[6] - i[5] for i in np_data]

        state = np.array([(i[1] * 3840, i[2] * 1920) for i in np_data]).reshape([19, 57, 100, 2])
        state_prob = np.array([(i[1], i[2]) for i in np_data]).reshape([19, 57, 100, 2])

        for idx, i in enumerate(np_data):
            if 99 == int(np_data[idx][0]):
                action = (0, 0)
            else:
                action = (np_data[idx + 1][1] - np_data[idx][1], np_data[idx + 1][2] - np_data[idx][1])
            actions.append(action)

        end = time.time()
        print("Y 걸린시간: {}초".format(end - start))

        print(np.shape(np_data))
        print(np.shape(actions))
        print(np.shape(frames))

        return np_data, np.array(actions)

    def plot_state_data(self, data):
        assert np.shape(data) == 2, 100
        for i in data:  # 57 2 100
            i = i.transpose()
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            ax.plot(i[0].tolist(), i[1].tolist(), range(100))
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


def draw_scanpath(fig, x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.plot(x, y, z)
    plt.title('test')
    plt.show()


if __name__ == '__main__':
    data, action = Sal360().load_sal360_dataset()
    print(np.shape(action.reshape(-1, 57, 100, 2)))