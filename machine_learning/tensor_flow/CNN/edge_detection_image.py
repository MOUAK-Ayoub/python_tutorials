import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image


def image_to_array(filename):
    im = Image.open('num3.jpg')
    image_gr = im.convert("L")
    arr = np.asarray(image_gr)
    imgplot = plt.imshow(arr)
    imgplot.set_cmap('gray')
    plt.show(imgplot)
    return arr

def edge_detection(im_array):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0], ])
    grad = signal.convolve2d(im_array, kernel, mode='same')
    fig, aux = plt.subplots(figsize=(10, 10))
    aux.imshow(grad, cmap='gray')
    plt.show()

if __name__ =='__main__':
    arr = image_to_array('num3.jpg')
    edge_detection(arr)

