import os
import csv
import numpy as np


class Sal360:
    def __init__(self):
        self.dataset_dir = "datasets"
        self.scanpath_H = os.path.join(self.dataset_dir, "Scanpaths_H", "Scanpaths")
        self.scanpath_HE = os.path.join(self.dataset_dir, "Scanpaths_HE")

    def read_scanpath_H(self):
        data = []
        print(sorted(os.listdir(self.scanpath_H)))
        for file in sorted(os.listdir(self.scanpath_H)):
            with open(os.path.join(self.scanpath_H, file)) as f:
                row = csv.reader(f)
                data.append(list(row)[1:])

        result = []
        for item in data:
            tmp = []
            for i in range(57):  # 각 영상당 57명
                tmp.append(item[0:100])
                del item[0:100]
            result.append(tmp)

        result = {"test": [i[45:] for i in result], "train": [i[:45] for i in result]}
        return result  # [19, 45, 100, 7], [19, 12, 100, 7]


if __name__ == '__main__':
    data = Sal360()
    data.read_scanpath_H()
