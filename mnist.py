import numpy as np
from my_resnet import dcn_resnet
import os
import tempfile
from keras import backend as K
import keras
import random
import matplotlib.pyplot as plt


mnist = keras.datasets.mnist

batch_size = 128
buffer_size = 10_000
num_classes = 10
steps_per_epoch = int(np.ceil((60_000 / float(batch_size))))
epochs = 5

(x_train, y_train), (x_test, y_test) = mnist.load_data()

data_in = (28, 28, 1)
# Create the dataset and its associated one-shot iterator.
# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# dataset = dataset.repeat()
# dataset = dataset.shuffle(buffer_size)
# dataset = dataset.batch(batch_size)
# iterator = dataset.make_one_shot_iterator()
# inputs, targets = iterator.get_next()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

model = dcn_resnet(data_in)
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=128, verbose=1,
                 validation_data=(x_test, y_test), epochs=epochs)
score = model.evaluate(x_test, y_test, verbose=0)

predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis=1)

test_labels = np.argmax(y_test, axis=1)

wrong_result = []

for n in range(0, len(test_labels)):
    if predicted_labels[n] != test_labels[n]:
        wrong_result.append(n)

samples = random.choices(population=wrong_result, k=16)

count = 0
nrows = ncols = 4

plt.figure(figsize=(12, 8))

for n in samples:
    count += 1
    plt.subplot(nrows, ncols, count)
    plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n])
    plt.title(tmp)

plt.tight_layout()
plt.show()
