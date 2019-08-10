import os
import csv
import numpy as np


def load_sal360_dataset():
    path = os.path.join("dataset", "H", "Scanpaths")
    data = []
    for file in sorted(os.listdir(path)):
        with open(os.path.join(path, file)) as f:
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
    load_sal360_dataset()
