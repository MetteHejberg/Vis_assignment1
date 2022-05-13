# import libraries
import os
import re

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img, img_to_array)
                                                  
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input, VGG16)

# scikit learn
from sklearn.neighbors import NearestNeighbors

import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt   
import matplotlib.image as mpimg

import argparse

# feature extraction function
def extract_features(img_path, model):
    # Define input image shape - remember we need to reshape
    input_shape = (224, 224, 3)
    # load image from file path
    img = load_img(img_path, target_size=(input_shape[0], 
                                          input_shape[1]))
    # convert to array
    img_array = img_to_array(img)
    # expand to fit dimensions
    expanded_img_array = np.expand_dims(img_array, axis=0)
    # preprocess image
    preprocessed_img = preprocess_input(expanded_img_array)
    # use the predict function to create feature representation
    features = model.predict(preprocessed_img)
    # flatten
    flattened_features = features.flatten()
    # normalize features
    normalized_features = flattened_features / norm(features)
    return normalized_features

# create VGG16 model
def mdl():
    model = VGG16(weights = "imagenet", 
              pooling ="avg", 
              include_top = False, 
              input_shape = (224, 224, 3))
    return model

# load the directory and create a list of its contents 
def load_paths(model):
    # set a directory path
    directory_path = os.path.join("..", "CDS-VIS", "flowers")
    # list the contents of the directory in order
    filenames = os.listdir(directory_path)
    # create an empty list
    joined_paths = []
    # for every file in filenames
    for file in filenames:
        # if the file doesn't end with .jpg
        if not file.endswith(".jpg"):
            # do nothing
            pass
        # if the file does end with .jpg
        else:
            # get the directory path and the file name
            input_path = os.path.join(directory_path, file)
            # and append to the list
            joined_paths.append(input_path)
    # sort the list in ascending order
    joined_paths = sorted(joined_paths)
    # create an empty list
    feature_list = []
    # for every file in joined_paths
    for input_file in joined_paths:
        # extract their features
        features = extract_features(input_file, model)
        # and append to the list
        feature_list.append(features)
    return joined_paths, feature_list

# k-nearest neighbors function that finds the most similar images to a target image
def knn(image, feature_list, model):
    # define parameters
    neighbors = NearestNeighbors(n_neighbors = 10, # find the 10 nearest neighbors 
                                 algorithm = "brute",
                                 metric = "cosine").fit(feature_list) # fit our features to out k-nearest neighbors algorithm
    # extract features from the user-defined image
    user_image = extract_features(os.path.join("..", "CDS-VIS", "flowers", image), model)
    # finds the closest images and their indices
    distances, indices = neighbors.kneighbors([user_image])
    # create an empty list
    idxs = []
    # let's get the most similar image
    for i in range(1,6):
        idxs.append(indices[0][i])
    return idxs, distances

# let's save the images with their distances and a csv with the file names 
def save_images_csv(image, joined_paths, idxs, distances):
    # create a  2x2 matrix for the images 
    f, ax = plt.subplots(2,2)
    # arrange the images in the matrix
    ax[0,0].imshow(mpimg.imread(os.path.join("..", "CDS-VIS", "flowers", image)))
    ax[0,1].imshow(mpimg.imread(joined_paths[idxs[0]]))
    ax[1,0].imshow(mpimg.imread(joined_paths[idxs[1]]))
    ax[1,1].imshow(mpimg.imread(joined_paths[idxs[2]]))
    # add the distances as text on the images 
    ax[0,1].text(0.5, 0.5, f"Distance:{distances[0][1]}", fontsize=7, ha="center")
    ax[1,0].text(0.5, 0.5, f"Distance:{distances[0][2]}", fontsize=7, ha="center")
    ax[1,1].text(0.5, 0.5, f"Distance:{distances[0][3]}", fontsize=7, ha="center")
    # save the figure
    f.savefig(os.path.join("out", "similar_images.jpg"))
    
    # create list of the most similar images from joined_paths
    similar = list((joined_paths[idxs[0]], joined_paths[idxs[1]], joined_paths[idxs[2]]))
    # clean the list with regex to only include the image names 
    similar = [re.sub(".*s\/", "", token) for token in similar]
    # create a dictionary of the data that should go into the data frame
    data = [{"Original Image": image, "Third": similar[2], "Second": similar[1], "First": similar[0]}]
    # make the dictionary into a data frame
    dframe = pd.DataFrame(data, columns = ["Original Image", "Third", "Second", "First"])
    # define an outpath
    outpath = os.path.join("out", "similar_images.csv")
    # save the data frame as a csv 
    dframe.to_csv(outpath, encoding = "utf-8")

def parse_args():
    # initialize argparse
    ap = argparse.ArgumentParser()
    # add command line parameters
    ap.add_argument("-i", "--image", required=True, help="the user-defined image")
    args = vars(ap.parse_args())
    return args

# let's run the code
def main():
    args = parse_args()
    model = mdl()
    joined_paths, feature_list = load_paths(model)
    idxs, distances = knn(args["image"], feature_list, model)
    save_images_csv(args["image"], joined_paths, idxs, distances)

if __name__ == "__main__":
    main()
    
                            
