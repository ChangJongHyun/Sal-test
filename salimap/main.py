from __future__ import division
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os, cv2, sys
import numpy as np
from salimap.config import *
from salimap.mlnet import ml_net_model, loss

if __name__ == '__main__':
    model = ml_net_model(img_cols=shape_c, img_rows=shape_r, downsampling_factor_product=10)
    sgd = SGD(lr=1e-3, decay=0.0005, momentum=0.9, nesterov=True)
    print("Compile ML-Net Model")
    model.compile(sgd, loss)

    print("Load weights ML-Net")
    model.load_weights('salimap/model/mlnet_salicon_weights.pkl')

    # print("Predict saliency maps for " + imgs_test_path)
    print(model.summary())
    # predictions = model.predict_generator(generator_test(b_s=1, imgs_test_path=imgs_test_path), nb_imgs_test)
    #
    # for pred, name in zip(predictions, file_names):
    #     original_image = cv2.imread(imgs_test_path + name, 0)
    #     res = postprocess_predictions(pred[0], original_image.shape[0], original_image.shape[1])
    #     cv2.imwrite(output_folder + '%s' % name, res.astype(int))
