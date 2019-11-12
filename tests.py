from data import Data_Loader
import matplotlib.pyplot as plt
import cv2


def vis_points(image, points, diameter=10):
    im = image.copy()

    for (x, y) in points:
        cv2.circle(im, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.imshow(im)

if __name__ == "__main__":
    dataset = Data_Loader('droso',10)
    dataset()
    for elem in dataset.data:
        image, keypoints = elem
        im2 = image.numpy().squeeze()
        keyp = keypoints.numpy()
        vis_points(im2, keyp)
        plt.show()