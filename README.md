## 1. Assignment 1 - Image search
Link to repository: https://github.com/MetteHejberg/Vis_assignment1

For this assignment, you will write a small Python program to compare image histograms quantitively using Open-CV and the other image processing tools you've already encountered. Your script should do the following:
- Take a user-defined image from the folder
- Calculate the "distance" between the colour histogram of that image and all of the others.
- Find which 3 image are most "similar" to the target image.
- Save an image which shows the target image, the three most similar, and the calculated distance score.
- Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

## 2. Methods 
This repository contains two scripts that find similarity between images

```color_histogram_comparison.py``` uses Open-CV to normalize images and extract their histograms. It also uses Open-CV to calcute distance scores based the histograms. It then finds the most similar images through the lowest distance scores. It lastly saves a 2x2 matrix with the target image and the three most similar images with their distances scores and a csv file with the filenames. This is a quick and easy way to get results, however it only takes colors into account, which can yield surprising results compared to how human judge similarity between images.

```k_nearest_neighbors``` uses a convolutional neural network to find similarity. The script extract features from the images using vgg16, passes the these features through k-nearest neighbors that returns distances scores and indices of the images. The most similar images are then found through the indices and plotted in a 2x2 matrix with the target image and the distance scores. Lastly, the script saves a csv file with the filenames. This approach is slower but perhaps also more like humans judge images to be similar. 

Get the data here: https://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html

## 3.1 Usage ```color_histogram_comparison.py```
To run the code you should:
- Pull this repository with this folder structure
- Place the images in the ```in``` folder
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/color_histogram_comparison.py -i "image_name.jpg"``` 
  - The outputs in ```out``` were created with ```python src/color_histogram_comparison.py -i "image_0005jpg"``` __fix because 005 looked so much better__

## 3.2 Usage ```k_nearest_neighbors.py```
To run the code you should:
- Pull this repository with this folder structure
- Place the images in the ```in``` folder
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/k_nearest_neighbors.py -i "image_name.jpg"```
  - The outputs in ```out``` were created with ```python src/color_histogram_comparison.py -i "image_0003.jpg"```

## 4. Discussion of Results 
Using the same user-defined image in both scripts makes it clear that k-nearest neighbors captures something closer to how humans judge images to be similar. The color histogram comparison was quite successfull on ```image_0005.jpg``` and less successfull on for examples ```image_0003.jpg```. k-nearest neightbors finds images of the same flower in very similar contexts and is even able to ignore the slightly different backgrounds


