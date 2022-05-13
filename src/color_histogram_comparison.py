# import libraries
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import re
import argparse

# create a function that loads all images 
def load_image_paths():
    # set a directory path
    directory_path = os.path.join("..", "CDS-VIS", "flowers")
    # get the file names
    filenames = os.listdir(directory_path) 
    # create an empty list
    joined_paths = []
    # for every file in filenames
    for file in filenames:
        # if the file doesn't end with .jpg 
        if not file.endswith(".jpg"):
            # do not append
            pass
        # if the file does end with .jpg
        else:
            # get directory path and the file name
            input_path = os.path.join(directory_path, file)
            # and append to the list
            joined_paths.append(input_path)
    # sort the list in ascending order        
    joined_paths = sorted(joined_paths)
    return joined_paths 

# create a function that extracts histograms from a single, user-defined image 
def hist_norm(image):
    # get the image
    img = os.path.join("..", "CDS-VIS", "flowers", image)
    # extract the histogram
    hist = cv2.calcHist([cv2.imread(img)],
                         [0,1,2],  
                         None, 
                         [8,8,8],  
                         [0,256, 0,256, 0,256])
    # normalize
    hist_norm_img = cv2.normalize(hist, hist, 0,255, cv2.NORM_MINMAX)
    return hist_norm_img

# do the same for all other images
def hist_norm_comp(joined_paths, hist_norm_img):
    # create an empty list
    ready_images = []
    # create another
    compare_images = []
    # for every file in joined_paths
    for file in joined_paths:
        # calculate the histograms
        image_hist = cv2.calcHist([cv2.imread(file)], [0, 1, 2], None, [8, 8, 8], [0,256, 0,256, 0,256])
        # normalize
        image_norm = cv2.normalize(image_hist, image_hist, 0,255, cv2.NORM_MINMAX)
        # and append to the list
        ready_images.append(image_norm)
    # for every file in ready_images    
    for image in ready_images:
        # compare the histograms with the user-defined images and get the distances
        image_comp = cv2.compareHist(hist_norm_img, image, cv2.HISTCMP_CHISQR)
        # and append the distances to the list
        compare_images.append(image_comp)
    # sort the list    
    compare_sort = sorted(compare_images)
    # find the most similar 
    similar_images = compare_sort[1:10]
    # get their scores
    p1 = similar_images[0]
    p2 = similar_images[1]
    p3 = similar_images[2]
    # create another empty list
    similar = []
    # get the indices of each of the distance scores from the unsorted list of distances scores
    img1 = compare_images.index(p1)
    img2 = compare_images.index(p2)
    img3 = compare_images.index(p3)
    # use this index to find the images in joined_paths
    image1 = joined_paths[img1]
    image2 = joined_paths[img2]
    image3 = joined_paths[img3]
    # and append the most similar images to the list
    similar.append([image1, image2, image3])
    return similar, similar_images

# a create a function that saves the images with their distances scores and a csv with the filenames of the images
def save_images_csv(similar, joined_paths, image, similar_images):
    # create an empty list
    final_images = []
    # for every token in similar
    for o in similar:
        # for every element in the token
        for i in o:
            # read the files as images
            sim_img = cv2.imread(os.path.join(i))
            # convert their colors so we can plot them
            rgb_image = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
            # append to the list
            final_images.append(rgb_image)
    # do the same for the user-defined image
    rgb_target = cv2.imread(os.path.join("..", "CDS-VIS", "flowers", image))
    rgb_target = cv2.cvtColor(rgb_target, cv2.COLOR_BGR2RGB)
    
    # let's plot the images
    # create a matrix of 2x2
    f, axarr = plt.subplots(2,2)
    # arrange each image in the matrix
    axarr[0,0].imshow(rgb_target)
    axarr[0,1].imshow(final_images[0])
    axarr[1,0].imshow(final_images[1])
    axarr[1,1].imshow(final_images[2])
    # add the distances scores as text
    axarr[0,1].text(0.5, 0.5, f"Distance:{similar_images[0]}", fontsize=7, ha="center")
    axarr[1,0].text(0.5, 0.5, f"Distance:{similar_images[1]}", fontsize=7, ha="center")
    axarr[1,1].text(0.5, 0.5, f"Distance:{similar_images[2]}", fontsize=7, ha="center")
    # create an outpath
    outpath = os.path.join("out", "similar_images.jpg")
    # save the figure
    f.savefig(outpath)
    
    # use regex to clean the file paths in similar to just include the image names
    similar = [[re.sub(".*s\/", "", i) for i in token] for token in similar]
    # create a dictionary of what should go into the data frame
    data = [{"Original Image": image, "Third": similar[0][2], "Second": similar[0][1], "First": similar[0][0]}]
    # convert the dictionary to a data frame
    dframe = pd.DataFrame(data)
    # create another outpath
    outpath2 = os.path.join("out", "similar_images.csv")
    # and save the data frame as a csv
    dframe.to_csv(outpath2, encoding = "utf-8")
    
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
    joined_paths = load_image_paths()
    hist_norm_img = hist_norm(args["image"])
    similar, similar_images = hist_norm_comp(joined_paths, hist_norm_img)
    save_images_csv(similar, joined_paths, args["image"], similar_images)
    
if __name__ == "__main__":
    main()
