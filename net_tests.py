from data import Data_Loader
import tensorflow as tf
from run import get_max_indices
import matplotlib.pyplot as plt
import cv2


def vis_points(image, points, diameter=10):
    im = image.copy()

    for (x, y) in points:
        cv2.circle(im, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.imshow(im)

if __name__ == "__main__":
    dataset = Data_Loader('droso',1)
    dataset()
    iterator = iter(dataset.data)
    
    image, keypoints = next(iterator)
    
    im2 = image.numpy().squeeze()
    #hm2 = keypoints.numpy().squeeze()[1]
    keyp = tf.squeeze(tf.map_fn(lambda x: tf.map_fn(get_max_indices,x), keypoints)).numpy()
    vis_points(im2, keyp)
    plt.show()