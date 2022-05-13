## 1. Assignment 1 - Image search
For this assignment, you will write a small Python program to compare image histograms quantitively using Open-CV and the other image processing tools you've already encountered. Your script should do the following:
- Take a user-defined image from the folder
- Calculate the "distance" between the colour histogram of that image and all of the others.
- Find which 3 image are most "similar" to the target image.
- Save an image which shows the target image, the three most similar, and the calculated distance score.
- Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

## 2. Methods 
This repository contains two scripts to find similarity between images

```color_histogram_comparison.py``` extracts histograms and calcutes distance scores based these. It is a quick and easy way to get results, however it only takes colors into account, which can yield surprising results 

```k_nearest_neighbors``` uses a convolutional neural network, VGG16 and k-nearest neighbors, to find similarity. This approach is slower but perhaps also more like humans judge images to be similar


## 3.1 Usage ```color_histogram_comparison.py```
To run the code you should:
- Pull this repository with this file structure
- Place the images in the ```in``` folder
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/color_histogram_comparison.py -i "image_name.jpg" 
  - The outputs in were created with ```python src/color_histogram_comparison.py -i "image_0003.jpg"

## 3.2 Usage ```k_nearest_neighbors.py```
To run the code you should:
- Pull this repository with this file structure
- Place the images in the ```in``` folder
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/k_nearest_neighbors.py -i "image_name.jpg" 
  - The outputs in were created with ```python src/color_histogram_comparison.py -i "image_0003.jpg"

## 4. Discussion of Results 
Using the same user-defined image in both scripts makes it clear that k-nearest neighbors captures something closer to how humans judge images to be similar.
