import tensorflow as tf
import numpy as np

char_rdir = ['h', 'e', 'l', 'o']
char_dic = {w: i for i, w in enumerate(char_rdir)}

ground_truth = [char_dic[c] for c in 'hello']
print(ground_truth[:-1])
x_data = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0]])

x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)

rnn_size = len(char_dic)
batch_size = 1
output_size = 4

rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size,
                                        activation=tf.tanh)

initial_state = rnn_cell.zero_state(batch_size, tf.float32)
print(initial_state)
x_split = tf.split(0, len(char_dic), x_data)
print(x_split)
outputs, state = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=x_split, initial_state=initial_state)
