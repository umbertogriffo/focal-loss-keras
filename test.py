import numpy as np
import tensorflow as tf
from keras import backend as K

from losses import categorical_focal_loss

import unittest


class TestFocalLoss(unittest.TestCase):

    def test_is_equal_to_categorical_cross_entropy_pixel_based(self):
        # Pixel-based batch size
        y_true = np.array([[0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]], dtype=np.float32)

        print("Pixel-based labelling")
        print("Data dimension as [batch_size (amount of pixels), one_hot_encoding of pixel label]")
        print(y_true.shape)

        print("categorical_cross_entropy")
        cce = tf.keras.losses.categorical_crossentropy
        cce_value = K.mean(cce(y_true, y_pred)).numpy()
        print(cce_value)

        print("focal_loss")
        cfl = categorical_focal_loss(alpha=[[1, 1, 1]], gamma=0.)
        cfl_value = cfl(y_true, y_pred).numpy()
        print(cfl_value)

        self.assertEqual(cce_value, cfl_value)

    def test_is_equal_to_categorical_cross_entropy_image_based(self):
        # Image based batch size
        y_true = np.array([[[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 0, 1], [0, 0, 1, 0]]],
                           [[[0, 1, 0, 0], [0, 1, 0, 0]], [[1, 0, 0, 0], [0, 0, 0, 1]]]])

        y_pred = np.array(
            [[[[0.8, 0.0, 0.2, 0.0], [0.0, 0.95, 0.0, 0.05]], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.0, 0.5, 0.0]]],
             [[[0.0, 0.6, 0.0, 0.4], [0.1, 0.80, 0.1, 0.00]], [[0.7, 0.0, 0.3, 0.0], [0.2, 0.0, 0.3, 0.5]]]],
            dtype=np.float32)

        print("Image-based labelling")
        print("Data dimension as [batch_size (amount of images), height, width, one_hot_encoding of pixel label]")
        print(y_true.shape)

        print("categorical_cross_entropy")
        cce = tf.keras.losses.categorical_crossentropy
        cce_value = K.mean(cce(y_true, y_pred)).numpy()
        print(cce_value)

        print("focal_loss")
        cfl = categorical_focal_loss(alpha=[[1, 1, 1, 1]], gamma=0.)
        cfl_value = cfl(y_true, y_pred).numpy()
        print(cfl_value)

        self.assertEqual(cce_value, cfl_value)

    def test_focal_loss_effectiveness_of_balancing(self):
        """ Test to verify the effectiveness of the weights between α balance categories.
        """

        y_true = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        y_pred = np.array([[0.3, 0.99, 0.8, 0.97, 0.85], [0.9, 0.05, 0.1, 0.09, 0]], dtype=np.float32)

        """
        Suppose we are dealing with a multi-class prediction problem with five outputs. 
        According to the above example, suppose our model predicts the first label poorly compared to other labels.
        """
        cfl_balanced = categorical_focal_loss(alpha=[[1, 1, 1, 1, 1]], gamma=0.)
        cfl_balanced_value = cfl_balanced(y_true, y_pred).numpy()
        print(cfl_balanced_value)

        """
        We use α to adjust the weight of the first label, and try to modify α to [[2, 1, 1, 1, 1]]. 
        The loss increases, indicating that the loss function has successfully enlarged the weight of 
        the first category, which will make the model pay more attention to the correct prediction of the first label.
        """
        cfl_unbalanced = categorical_focal_loss(alpha=[[2, 1, 1, 1, 1]], gamma=0.)
        cfl_unbalanced_value = cfl_unbalanced(y_true, y_pred).numpy()
        print(cfl_unbalanced_value)

        self.assertGreater(cfl_unbalanced_value, cfl_balanced_value)


