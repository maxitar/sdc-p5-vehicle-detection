from glob import glob
import numpy as np
import os
import cv2
from skimage.feature import hog

def read_images(train_split = 0.7):
    """
    Reads image names and separates them in training and test sets.
    The first 'train_split' fraction of images from each folder goes
    into the training set and the rest go into the test set.
    """
    vehicles_train = []
    vehicles_test = []
    for path in os.listdir('../training_set/vehicles/'):
        path = '../training_set/vehicles/' + path
        if os.path.isdir(path):
            all_files = glob(path+'/*.png')
            num_train = int(len(all_files)*train_split)
            vehicles_train.extend(all_files[:num_train])
            vehicles_test.extend(all_files[num_train:])
        #train1 = glob('../training_set/vehicles/GTI_far/*.png')
    non_vehicles_train = []
    non_vehicles_test = []
    for path in os.listdir('../training_set/non-vehicles/'):
        path = '../training_set/non-vehicles/' + path
        if os.path.isdir(path):
            all_files = glob(path+'/*.png')
            num_train = int(len(all_files)*train_split)
            non_vehicles_train.extend(all_files[:num_train])
            non_vehicles_test.extend(all_files[num_train:])

    return vehicles_train, vehicles_test, non_vehicles_train, non_vehicles_test

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec, block_norm='L2')
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec, block_norm='L2')
        return features

def get_features(img, orientations, pixels_per_cell, cells_per_block):
    img_ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ch1 = img_ycc[:,:,0]
    ch2 = img_ycc[:,:,1]
    ch3 = img_ycc[:,:,2]
    hog_ch1 = get_hog_features(ch1, orientations, pixels_per_cell, cells_per_block)
    hog_ch2 = get_hog_features(ch2, orientations, pixels_per_cell, cells_per_block)
    hog_ch3 = get_hog_features(ch3, orientations, pixels_per_cell, cells_per_block)
    hog_features = np.concatenate((hog_ch1.ravel(), hog_ch2.ravel(), hog_ch3.ravel()))

    use_two_colors = False
    if use_two_colors:
        img_luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        ch1 = img_luv[:,:,0]
        ch2 = img_luv[:,:,1]
        ch3 = img_luv[:,:,2]
        hog_ch1 = get_hog_features(ch1, orientations, pixels_per_cell, cells_per_block)
        hog_ch2 = get_hog_features(ch2, orientations, pixels_per_cell, cells_per_block)
        hog_ch3 = get_hog_features(ch3, orientations, pixels_per_cell, cells_per_block)
        hog_features = np.append(hog_features, np.concatenate((hog_ch1.ravel(), hog_ch2.ravel(), hog_ch3.ravel())))

    return hog_features

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def get_labeled_bboxes(labels):
    boxes = []
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        boxes.append(bbox)
    # Return the image
    return boxes
