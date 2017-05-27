
# Vehicle Detection Project



The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


##  Data 
The data used for this project contains image postage stamps from vehicles and non-vehicles which can be obtained from [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) repectively.
Each image if 64x64 in 3 color channels in 'png' format with pixel values between 0-1. In total I used 8792 vehicle images and 8968 non-vehicle images, which is more or less a balanced distribution. The figure below shows a few examples of each class. 

-- add image

These images are used to train the classifier that predicts the image class (car = 1 and non-car = 0).
The figure below shows the distribution of classes in the dataset:

-- add image

In the following sections I explain how I created feature vectors from this dataset to train and test the classification model.

## Features

### HOG Features
### Color Features
### Color Space
### Spatial Binning


## Feature Extraction
The final feature vector is combination of all the individual feature vectors. This means that all individual feature vectors are appended together to create one long vector of 6156 elements per image. This includes HOG features in all three channels, color features (with 32 bins) and also spatial features (16,16). I chose the HLS color space and specifically the S channel as with some trial and error seemed to be contributing to the model accuracy the most.Also with hog transform, I used 9 orientations, 8 pixels per cell and 2 cells per block.

## Data Preparation 
There are a few other steps before the feature vector is ready to be used to train the classifier.
### Scaling 
The first step after building the feature vector is to normalize it so that it has mean = 0 and variance = 1. This step is done using ```StandardScaler().fit(X)``` function from sklearn.
### Train/Test Split
The next is to shuffle the data so that the order at which the classifier recieves the data is random. I then split the data to training and testing sets with ratio of 80:20. Both the shuffle and split is done using ```train_test_split()``` function in sklearn.
The figure below shows the number of training (blue) and testing (orange) classes in the dataset.

-- add image

As it can be seen both training and testing datasets are pretty much balanced between the two classes.

## Classification Model
The next step is training the classification model. I use a linear SVM model as discussed in the lectures. The number of training samples are 14208.
The prediction accuary on the test dataset (3552) is 99.2%.
I have played with the elements of feature vector to improve the model performace. By including all channels rather than just one and changing color space I improved the accuracy on the test sample from 94% to 99%. The length of the feature vector nearly triples and the predictions take longer to process, however the accuracy gain is very valuable at this stage, given that the goal is to control the number of false positive detections.

## Windowed Search
 


I choose a different size window for different regions of the image, since cars in the distance appear to be smaller, so smaller search windows are only applied to distant regions and larger windows to regions closer to the driver.
The 5 window sizes I chose are: 70, 120, 150, 180, 240 pixels per side, and the region of the image they are applied in y are [410, 480],[400, 520],[400, 550],[380,550],[400,640] respectively.

The figure below shows the overlapping windows of variious sizes the way they will be overlaid on each image. I used 75% overlap between each window pair.  

--- add image

With this configration, the total number of windows applied to each image is: 160.
The image gets scanned by each window and that patch is passed to the classifier to predict yes,no for vehicle detection. If vehicle is detected on the patch, all pixel values within that patch will be increased by 1. At the end of the search each pixel has been visited at least once by the window search. In the next section I discuss the search hot and cold zones in more detail.

-- add overlapping window image


## Heatmap and Thresholding


## Video

## Future Improvements








###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
 ### Future Improvements
 I think using the window size as a function of poistion in y axis, is a smarter way to reduce both number of total windows and also false positive detections, since only larger windows will be allowed near the bottom of image, so the chances of small patches near the bottom  being classified as car wouldbe reduced.
