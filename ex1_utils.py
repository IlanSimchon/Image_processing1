"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List


import numpy as np
import cv2
import  matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 212036396


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)

    if img is None:
        print("error in path")
    if representation == LOAD_GRAY_SCALE:
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == LOAD_RGB:
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Illegal representation")
        return
    scaling = (new_img - np.min(new_img)) / (np.max(new_img) - np.min(new_img))

    return scaling




def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename , representation)

    plt.imshow(img ,cmap='gray')

    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    mat = np.array([[0.299 , 0.587 , 0.114] ,
                    [0.596 , -0.275 , -0.321] ,
                    [0.212 , -0.523 , 0.311]])
    rows, cols, dim = imgRGB.shape

    img = imgRGB.reshape(-1  , 3)
    YIQ = np.dot(mat , img.T)

    YIQ = YIQ.T.reshape(rows, cols, dim)

    return YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    inv_mat = np.linalg.inv(mat)

    rows, cols, dim = imgYIQ.shape
    img = imgYIQ.reshape(-1 , 3)
    rgb = np.dot(inv_mat , img.T)

    rgb = rgb.T.reshape(rows, cols , dim)

    return rgb


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # check if it rgb or gray scale
    is_rgb = False
    original_img = imgOrig
    if len(imgOrig.shape) == 3:
        is_rgb = True
        yiq = transformRGB2YIQ(imgOrig)
        imgOrig = yiq[:,:,0]

    # normalize the image from 0-1 to 0-255 and flat the image
    imgOrig = cv2.normalize(imgOrig,None , 0,255 , cv2.NORM_MINMAX).astype('uint8')
    img_ravel = imgOrig.ravel()

    # making histograma
    histORG , bins = np.histogram(img_ravel , bins = 256)

    # Calculate Cumulative Sum
    cumSum = np.cumsum(histORG)

    # create Look up table
    norm_cumSum = cumSum / cumSum.max() * 255
    LUT = np.ceil(norm_cumSum).astype('uint8')

    # replace the colors by the LUT
    new_data = np.zeros_like(imgOrig)
    for idx in range(256):
        new_data[imgOrig == idx] = LUT[idx]

    # normalize, calculate histogram and cum sum
    eqImg = cv2.normalize(new_data, None, 0, 255,  cv2.NORM_MINMAX).astype('uint8')
    eqHist, eqBins = np.histogram(eqImg, bins=256)


    # normalize the image back
    eqImg = eqImg / 255

    # set the Y of YIQ when the original image is RGB
    if is_rgb is True:
        yiq[:,:,0] = eqImg
        eqImg = transformYIQ2RGB(yiq)

    return eqImg, histORG, eqHist

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # check if it rgb or gray scale
    is_rgb = False
    if len(imOrig.shape) == 3:
        is_rgb = True
        yiq = transformRGB2YIQ(imOrig)
        imOrig = yiq[:, :, 0]

    # normalize and create histogram
    imOrig = cv2.normalize(imOrig, None , 0,255 , cv2.NORM_MINMAX)
    hist,binso = np.histogram(imOrig , bins=256)

    # making borders using np.linspace
    borders = np.linspace(0,255,nQuant+1 , dtype=int)

    # the lists we will return
    images = []
    errors = []

    # This variable will count how many times the MSE came out the same
    count_mse = 0

    # Our main loop that run nIter times (or less, according to mse)
    for iter in range(nIter):
        meanPix = []

        # find the mean of each part
        for idx in range(0,len(borders)-1):
            sum = 0
            count = 0
            for i in range(borders[idx] , borders[idx+1]):
                sum += hist[i] * i
                count += hist[i]
            if count != 0:
                meanPix.append((sum / count).astype('int'))
            else:
                meanPix.append(((borders[idx] + borders[idx+1])/2).astype('int'))
        # replace the borders by the meanPix
        for i in range(0, len(borders)-2):
            borders[i+1] = ((meanPix[i]+meanPix[i+1])/2)

        # making image with just nQuant pixels
        tempImg = np.zeros_like(imOrig)
        for i in range(0, nQuant):
            tempImg[imOrig > borders[i]] = meanPix[i]

        # calculate the mse of this iteration
        mse = np.sqrt((imOrig - tempImg) ** 2).mean()

        if len(errors) > 1 and mse == errors[-1]:
            count_mse += 1
        else:
            count_mse = 0

        # If the mse is the same in the last 7 iterations, we will exit the loop
        if count_mse == 7:
            break
        errors.append(mse)

        # set the Y of YIQ when the original image is RGB
        if is_rgb:
            yiq[:,:,0] = tempImg / 255
            images.append(transformYIQ2RGB(yiq))
        else:
            images.append(tempImg / 255)

    return images, errors
