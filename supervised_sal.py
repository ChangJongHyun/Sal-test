import numpy as np
import os
import tempfile
import random
import matplotlib.pyplot as plt
from dataset import Sal360
from custom_env.gym_my_env.envs.viewport import Viewport
from resnet_tf import ResNet50
import tensorflow as tf
import cv2


# 1. supervided learning cnn + rnn
# 2. RL --> object detection --> tracking (saliency) --> reward을
# 3. IRL --> reward function
# action 정의 매프레임? 묶어서?
# multimodal
# 음향 어떻게? 3D JS?
class Supervised:
    def __init__(self, data_format):
        self.model = ResNet50(data_format, include_top=False)
        self.path = 'model/supervised'

    def learn(self, epochs):
        pass

    def save_model(self, sess):
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path=os.path.join(self.path, 'model.ckpt'))
        print('Model saved in path : {}'.format(self.path + 'model.ckpt'))

    def load_model(self, sess):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(self.path, 'model.ckpt'))
            print("Model restored")


def random_batch(batch_size, data_format):
    shape = (3, 224, 224) if data_format == 'channels_first' else (224, 224, 3)
    shape = (batch_size,) + shape

    num_classes = 1000
    images = tf.random_uniform(shape)
    labels = tf.random_uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    one_hot = tf.one_hot(labels, num_classes)

    return images, one_hot


if __name__ == '__main__':
    sign_ary = [[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0., -1.], [-1., 0.], [-1., -1.], [-1., 1.], [1., -1.]]

    sal360 = Sal360()
    train, test = sal360.load_sal360_dataset()

    width = 3840
    height = 1920

    data_format = 'channels_last'

    view = Viewport(width, height)
    images = []
    # model = ResNet50(data_format, include_top=False)

    # end - start 가 6이면 뒤에꺼 1개 프레임 짤라
    dataX = []
    # output = model(images, trainable=True)
    for video, data in zip(sorted(os.listdir('sample_videos/3840x1920/')), dataset):
        # data --> [57, 100, 7]
        cap = cv2.VideoCapture(os.path.join('sample_videos/3840x1920/', video))
        for scan in data:
            c_idx = 0
            idx = 0
            while True:
                ret, frame = cap.read()

                if c_idx < 100 and idx % 5 == 0:
                    w = float(scan[c_idx][2]) * width
                    h = float(scan[c_idx][1]) * height
                    view.set_center(np.array([w, h]))
                    dataX.append(view.center)
                    c_idx += 1

                if ret:
                    frame = view.get_view(frame)
                    frame = cv2.resize(frame, (224, 224))
                    cv2.imshow('test', frame)
                    idx += 1
                    print(np.shape(dataX))
                else:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
