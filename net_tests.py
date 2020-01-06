from data import Data_Loader
import tensorflow as tf
from run import get_max_indices
import matplotlib.pyplot as plt
import cv2

import numpy as np

def vis_points(image, points, diameter=5):
    im = image.copy()
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

    for (y, x) in points:
        cv2.circle(im, (int(x), int(y)), diameter, (255, 0, 0), -1)

    plt.imshow(im)

if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    dataset = Data_Loader('droso',8)
    dataset()
    iterator = iter(dataset.data)
    for _ in range(5):
            
        image, keypoints = next(iterator)
        
        im2 = image[0].numpy().squeeze()
        hm2 = np.sum(keypoints[0].numpy().squeeze(),axis=0)
        keyp = tf.map_fn(lambda x: tf.map_fn(get_max_indices,x), keypoints)[0].numpy()
        vis_points(im2, keyp)
        plt.show()