import os
import csv
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import math

def preprocessing(idx, i):
    if idx == 99:
        return 0, 0
    else:
        return


class Sal360:
    def __init__(self):
        # self.dataset = self.load_sal360_dataset()
        pass

    def load_sal360_dataset(self):

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

        print(np.shape(np_data))
        print(actions[0])
        print(np.shape(frames))

        # 19 57 2, 100

        #  plot trajectory
        # for i in state[0]:  # 57 2 100
        #     i = i.transpose()
        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #     ax.set_xlabel('X axis')
        #     ax.set_ylabel('Y axis')
        #     ax.set_zlabel('Z axis')
        #     ax.plot(base[0], base[1], range(100))
        #     ax.plot(i[0].tolist(), i[1].tolist(), range(100))
        #     plt.title("test")
        #     plt.show()
        base = state[0][0]  # 100, 2
        print(base.shape)
        base_T = state[0][0].transpose()  # 2, 100
        base_prob = state_prob[0][0]
        base_prob_T = state_prob[0][0].transpose()

        def l2norm(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        # for i, ii in zip(state_prob[0], state[0]):  # 57 2 100  # KLdivergence, L2norm
        #     i_T = i.transpose()
        #     distance = [l2norm(p1, p2) for p1, p2 in zip(base, ii)]
        #
        #     kld_x = stats.entropy(pk=base_prob_T[0], qk=i_T[0])
        #     kld_y = stats.entropy(pk=base_prob_T[1], qk=i_T[1])
        #
        #     print("KL divergence (x, y) : ({}, {})".format(kld_x, kld_y))
        #     print("avg L2 norm : {}".format(min()))

        # for item in data:
        #     tmp = []
        #     for i in range(57):  # 각 영상당 57명
        #         tmp.append(item[0:100])
        #         del item[0:100]
        #     result.append(tmp)
        # [19, 45, 100, 7], [19, 12, 100, 7]
        # input --> (x, y)를 한 view
        # output --> one hot encoding된 value
        return [i[:45] for i in result], [i[45:] for i in result]  # (19, 57, 100, 7), (19, 57, 100)


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
    sal = Sal360()
    train, test = sal.load_sal360_dataset()
    #
    # train, test = np.array(train), np.array(test)
    # train.reshape()
    # print(np.shape(train))
    # width = 3840
    # height = 1920
    #
    fig = plt.figure()
    #
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    #
    # for idx, data in enumerate(ddd['train'][2]):
    #     transpose = np.transpose(data)
    #
    #     x = transpose[0]
    #     y = transpose[1]
    #     z = transpose[2]
    #
    #     x = list(map(lambda i: int(i), x))
    #     y = list(map(lambda i: float(i) * width, y))
    #     z = list(map(lambda i: float(i) * height, z))
    #     ax.plot(y, z, x)
    #
    # plt.title('test')
    # plt.show()
