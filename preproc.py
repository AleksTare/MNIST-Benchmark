import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix


def remove_constant_pixels(pixels_df):
    """Removes from the images the pixels that have a constant intensity value,
    either always black (0) or white (255)
    Returns the cleared dataset & the list of the removed pixels (columns)"""

    # Remove the pixels that are always black to compute faster
    changing_pixels_df = pixels_df.loc[:]
    dropped_pixels_b = []

    # Pixels with max value =0 are pixels that never change
    for col in pixels_df:
        if changing_pixels_df[col].max() == 0:
            changing_pixels_df.drop(columns=[col], inplace=True)
            dropped_pixels_b.append(col)
    print("Constantly black pixels that have been dropped: {}".format(dropped_pixels_b))

    # Same with pixels with min=255 (white pixels)
    dropped_pixels_w = []
    for col in changing_pixels_df:
        if changing_pixels_df[col].min() == 255:
            changing_pixels_df.drop(columns=[col], inplace=True)
            dropped_pixels_w.append(col)
    print("\n Constantly white pixels that have been dropped: {}".format(dropped_pixels_w))

    #print(changing_pixels_df.head())
    print("Remaining pixels: {}".format(len(changing_pixels_df.columns)))
    print("Pixels removed: {}".format(784 - len(changing_pixels_df.columns)))


    return changing_pixels_df, dropped_pixels_b + dropped_pixels_w

def get_bounding_box(grad, threshold):
    """Get the bounding box around a digit, expressed as x and y coordinates
    """
    exceeds_threshold = np.absolute(grad) > threshold
    diff = np.diff(exceeds_threshold)
    boundaries = []
    for i, (e, d) in enumerate(zip(exceeds_threshold, diff)):
        breaks = np.where(d)[0]
        assert breaks.shape[0] > 0
        if e[0]:
            breaks = np.array([0, breaks[0]])
        if e[-1]:
            breaks = np.array([breaks[0], d.shape[0]])
            breaks
        breaks[0] = breaks[0] + 1
        boundary = (breaks[0], breaks[-1])
        boundaries.append(boundary)
    return np.array(boundaries)


def get_data_to_box(df, threshold=.1):
    """get bounding boxes for all digits in dataset df
    """
    z_grad, y_grad, x_grad = np.gradient(df)
    y_grad_1d = y_grad.sum(axis=2)
    x_grad_1d = x_grad.sum(axis=1)

    y_bounds = get_bounding_box(y_grad_1d, threshold)
    x_bounds = get_bounding_box(x_grad_1d, threshold)

    return np.hstack([x_bounds, y_bounds])

def reshape_to_img(df):
    """Reshape 1d images to 2d
    """
    m = df.shape[0]
    i = np.int(np.sqrt(df.shape[1]))
    return df.reshape((m, i, i))