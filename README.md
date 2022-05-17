## 1. Assignment 1 - Image search
For this assignment, you will write a small Python program to compare image histograms quantitively using Open-CV and the other image processing tools you've already encountered. Your script should do the following:
- Take a user-defined image from the folder
- Calculate the "distance" between the colour histogram of that image and all of the others.
- Find which 3 image are most "similar" to the target image.
- Save an image which shows the target image, the three most similar, and the calculated distance score.
- Save a CSV which has one column for the filename and three columns showing the filenames of the closest images in descending order

## 2. Methods 
This repository contains two scripts that find similarity between images

```color_histogram_comparison.py``` uses Open-CV to normalize images and extract their histograms. It also uses Open-CV to calcute distance scores based the histograms. It then finds the most similar images through the lowest distance scores. It lastly saves a 2x2 matrix wit the target image and the three most similar images with their distances scores. This is a quick and easy way to get results, however it only takes colors into account, which can yield surprising results compared to how human judge similarity between images.

```k_nearest_neighbors``` uses a convolutional neural network to find similarity. The script extract features from the images using vgg16, passes the these features through k-nearest neighbors that returns distances scores and indices of the images. The most similar images are then found through the indices and plotted in a 2x2 matrix with the target image and the distance scores. This approach is slower but perhaps also more like humans judge images to be similar. 


## 3.1 Usage ```color_histogram_comparison.py```
To run the code you should:
- Pull this repository with this file structure
- Place the images in the ```in``` folder
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/color_histogram_comparison.py -i "image_name.jpg"``` 
  - The outputs in were created with ```python src/color_histogram_comparison.py -i "image_0003.jpg"```

## 3.2 Usage ```k_nearest_neighbors.py```
To run the code you should:
- Pull this repository with this file structure
- Place the images in the ```in``` folder
- Install the packages mentioned in ```requirements.txt```
- Set your current working directory to the level above ```src```
- Write in the command line: ```python src/k_nearest_neighbors.py -i "image_name.jpg"```
  - The outputs in were created with ```python src/color_histogram_comparison.py -i "image_0003.jpg"```

## 4. Discussion of Results 
Using the same user-defined image in both scripts makes it clear that k-nearest neighbors captures something closer to how humans judge images to be similar. Not only do the program choose different images to be the most similar, but they are also vastly different. In general, the color histogram comparison does quite well on the ```image_0003.jpg``` however the second most similar is a completely different flower in another color than the user-defined image. k-nearest neightbors finds images of the same flower in very similar contexts and is even able to ignore the slightly different backgrounds


