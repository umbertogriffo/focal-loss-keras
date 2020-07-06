import numpy as np
import tensorflow as tf
from keras import backend as K

from losses import categorical_focal_loss

import unittest


class TestFocalLoss(unittest.TestCase):

    def test_is_equal_to_categorical_cross_entropy_pixel_based(self):
        # Pixel-based batch size
        y_true = np.array([[0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

        print("Pixel-based labelling")
        print("Data dimension as [batch_size (amount of pixels), one_hot_encoding of pixel label]")
        print(y_true.shape)

        print("categorical_cross_entropy")
        cce = tf.keras.losses.categorical_crossentropy
        cce_value = K.mean(cce(y_true, y_pred)).numpy()
        print(cce_value)

        print("focal_loss")
        cfl = categorical_focal_loss(gamma=0., alpha=1.)
        cfl_value = cfl(y_true, y_pred).numpy()
        print(cfl_value)

        self.assertEqual(cce_value, cfl_value)

    def test_is_equal_to_categorical_cross_entropy_image_based(self):
        # Image based batch size
        y_true = np.array([[[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 0, 1], [0, 0, 1, 0]]],
                           [[[0, 1, 0, 0], [0, 1, 0, 0]], [[1, 0, 0, 0], [0, 0, 0, 1]]]])

        y_pred = np.array(
            [[[[0.8, 0.0, 0.2, 0.0], [0.0, 0.95, 0.0, 0.05]], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.0, 0.5, 0.0]]],
             [[[0.0, 0.6, 0.0, 0.4], [0.1, 0.80, 0.1, 0.00]], [[0.7, 0.0, 0.3, 0.0], [0.2, 0.0, 0.3, 0.5]]]])

        print("Image-based labelling")
        print("Data dimension as [batch_size (amount of images), height, width, one_hot_encoding of pixel label]")
        print(y_true.shape)

        print("categorical_cross_entropy")
        cce = tf.keras.losses.categorical_crossentropy
        cce_value = K.mean(cce(y_true, y_pred)).numpy()
        print(cce_value)

        print("focal_loss")
        cfl = categorical_focal_loss(gamma=0., alpha=1.)
        cfl_value = cfl(y_true, y_pred).numpy()
        print(cfl_value)

        self.assertEqual(cce_value, cfl_value)
