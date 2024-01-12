import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def cos_window(sz):
    """
    width, height = sz
    j = np.arange(0, width)
    i = np.arange(0, height)
    J, I = np.meshgrid(j, i)
    cos_window = np.sin(np.pi * J / width) * np.sin(np.pi * I / height)
    """

    cos_window = np.hanning(int(sz[1]))[:, np.newaxis].dot(np.hanning(int(sz[0]))[np.newaxis, :])
    return cos_window





def gaussian2d_rolled_labels(sz,sigma):
    w,h=sz
    xs, ys = np.meshgrid(np.arange(w)-w//2, np.arange(h)-h//2)
    dist = (xs**2+ys**2) / (sigma**2)
    labels = np.exp(-0.5*dist)
    labels = np.roll(labels, -int(np.floor(sz[0] / 2)), axis=1)
    labels=np.roll(labels,-int(np.floor(sz[1]/2)),axis=0)
    return labels

