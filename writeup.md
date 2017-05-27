
# Vehicle Detection Project



The goals / steps of this project are the following:
The goal of this project was to design a pipeline (computer vision based and not deep learning), to detect cars on the video frames.
In order to achive this goal the following steps were taken:
* Extract features from postage stamp images of cars and non cars to train a classifier.
* Set and train a classifier and tune the feature vector to optimize the accuacy of the classifier on test sample.
* Apply a search window on the video frames and for each window, use the trained classifier to predict whether or not the window includes a car.
* Setup a heatmap technique to combine the detection across all windows on the image frame to generate a car detection zone.
* Apply the pipeline to the video stream.
I will go over each and every one of the steps above is more details with visual examples in the following sections. The code for the project is split into two files:
'lesson_functions.py' contains the helper functions provided by Udacity and 'Vehicle_Detection.ipynb' contains the pipeline and the model training and video processing.

[//]: # (Image References)
[box_heatmap_thresh]: ./writeup_images/box_and_thresh_heatmap.png
[data_sample]: ./writeup_images/car_noncar_examples.png
[car_detection]: ./writeup_images/car_detection.png

[all_and_hot_windows]: ./writeup_images/all_and_hot_windows.png
[all_and_hot_heat]: ./writeup_images/all_and_hot_windows_heat.png

[class_dist]: ./writeup_images/class_distr.png
[color_f]: ./writeup_images/color_features.png
[HLS]: ./writeup_images/HLS_car.png
[HSV]: ./writeup_images/HSV_car.png
[LUV]: ./writeup_images/LUV_car.png
[hog_f]: ./writeup_images/hog_features.png
[hog_im]: ./writeup_images/hog_image.png
[spatial_f]: ./writeup_images/spatial_features.png

[train_test_dist]: ./writeup_images/train_test_class_distr.png
[YUV]: ./writeup_images/YUV_car.png
[YCC]: ./writeup_images/YCrCb_car.png

[video1]: ./project_video.mp4


##  Data 
The data used for this project contains image postage stamps from vehicles and non-vehicles which can be obtained from [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) repectively.
Each image is 64x64 in 3 color channels in 'png' format with pixel values between 0-1. In total I used 8792 vehicle images and 8968 non-vehicle images, which is more or less a balanced distribution. The figure below shows a few examples of each class. 

![alt text][data_sample]

These images are used to train the classifier that predicts the image class (car = 1 and non-car = 0).
The figure below shows the distribution of classes in the dataset:

![alt text][class_dist]


In the following sections I explain how I created feature vectors from this dataset to train and test the classification model.

## Features

### HOG (Histogram of Oriented Gradients) Features
One of the features I found important in improving the model accuracy if the HOG features, which I used in all 3 channls for HLS colorspace.
I visualized various orient, orient, pix_per_cell and cell_per_block value and for the final model settled at values of 9,8 and 2 respectively, such that I achieve good performance without making the feature vector too long.
In the figure below I show and example of hog tranformation on the first channel of RGB image (R):

![alt text][hog_im]
As it can be seen hog transformation provides a good indication of the overall shape of the object and by tuning parameters we can arrive at reasonable distinction between car and non-car shapes.
In the next figure I show the corresponding hog feature vector for the above image:

![alt text][hog_ft]



### Color Features
One the elements of the feature vector is the histogram of pixel value intentisity in the three color channels.
Here is an example of color features for car and non-car images:

![alt text][color_f]

### Color Space
The choice of color space can make an impact on the classification accuacy. The differece between car and non-car images are more visible in certain colorspace channels than others. Here I provide examples of such differences accorss a series of colorspaces.
#### HSV:
![alt text][HSV]
#### LUV:
![alt text][LUV]
#### YUV:
![alt text][YUV]
#### YCrCb:
![alt text][YCC]


At the end I chose to go with the HLS colorspace and more specificcally the S channel, as it seems to pick up the color saturation of the cars over the less saturated background well.
Here is an example of S channel in HLS color space on a sample car and non-car image:
#### HLS:
![alt text][HLS]

### Spatial Binning
Another feature is the spatial binning of the original image. For the final model, I resize each image to 16x16 and then convert it to a vector, which then gets added to the combined feature vector.
In the figure below I show an example of 32x32 resize, so the length of the feature vector in each channel is 32x32=1024 and for all 3 channels back to back the length of feature vector from spatial binning is 3x1024=3072 which is displayed below:
![alt text][spatial_f]


## Feature Extraction
The final feature vector is combination of all the individual feature vectors. This means that all individual feature vectors are appended together to create one long vector of 6156 elements per image. This includes HOG features in all three channels, color features (with 32 bins) and also spatial features (16,16). I chose the HLS color space and specifically the S channel as with some trial and error seemed to be contributing to the model accuracy the most.Also with hog transform, I used 9 orientations, 8 pixels per cell and 2 cells per block.

## Data Preparation 
There are a few other steps before the feature vector is ready to be used to train the classifier.
### Scaling 
The first step after building the feature vector is to normalize it so that it has mean = 0 and variance = 1. This step is done using ```StandardScaler().fit(X)``` function from sklearn.
### Train/Test Split
The next is to shuffle the data so that the order at which the classifier recieves the data is random. I then split the data to training and testing sets with ratio of 80:20. Both the shuffle and split is done using ```train_test_split()``` function in sklearn.
The figure below shows the number of training (blue) and testing (orange) classes in the dataset.

![alt text][train_test_dist]

As it can be seen both training and testing datasets are pretty much balanced between the two classes.

## Classification Model
The next step is training the classification model. I use sklearn's linear SVM model ```SVM()```as discussed in the lectures. The number of training samples are 14208.
The prediction accuary on the test dataset (3552) is 99.2%.
I have played with the elements of feature vector to improve the model performace. By including all channels rather than just one and changing color space I improved the accuracy on the test sample from 94% to 99%. The length of the feature vector nearly triples and the predictions take longer to process, however the accuracy gain is very valuable at this stage, given that the goal is to control the number of false positive detections.

## Windowed Search
 
I choose a different size window for different regions of the image, since cars in the distance appear to be smaller, so smaller search windows are only applied to distant regions and larger windows to regions closer to the driver.
The 5 window sizes I chose are: 70, 120, 150, 180, 240 pixels per side, and the region of the image they are applied in y are [410, 480],[400, 520],[400, 550],[380,550],[400,640] respectively.

The figure below shows the overlapping windows of variious sizes the way they will be overlaid on each image. I used 75% overlap between each window pair.  

![alt text][all_and_hot_windows]

With this configration, the total number of windows applied to each image is: 160.
The image gets scanned by each window and that patch is passed to the classifier to predict yes,no for vehicle detection. If vehicle is detected on the patch, all pixel values within that patch will be increased by 1. At the end of the search each pixel has been visited at least once by the window search. In the next section I discuss the search hot and cold zones in more detail.


![alt text][all_and_hot_heat]
One a side note, each frame from the video is in 'jpg' format, while the classifier has been trained on 'png' images, so the video frame images are scaled down to 0-1 before being windowed and fed into the classifier prediction.

## Heatmap and Thresholding
As shown in the figure above, due to window size and overlap amount, various pixels in the image are sweeped by the windows a different number of times. I generage a heatmap of all the window sweeps per pixel, which I consider as cold and hot pixels as shown in the left panel.
I also generate a hot heatmap which only has positive values for pixels for which the classification has returned a positive car detection, as shown in the right panel.
In the next step, I multiply the two maps which is equivalant to weighting the hot pixels by the number of times that pixel has been visited by search windows. 
Next I threshold the resulting image, such that only pixels with high values (due to multiple posivitve detections) are kept and using labels from ```scipy.ndimage.measurements.label()``` to obtain an image of labels. I then plot the boxes on the original image which can be seen below:
![alt text][car_detection]

As it can be seen the pipeline correctly identifies the car in the image. The next step is to apply the pipleline to video frames provided for this project. My final video file is *detecting_cars_all.mp4* in the same repo.
The goal of the project is reached and cars are detetcted on the video stream with minimal number of false detections.
That being said there is always room for improvement. Below I will breifly describe a few ideas I would like to try next.

## Future Improvements
To improve the car detection,  I would like to focus on reducing the number of false positive detections further. One way would be to appy a mask on the image an disallow window search to search the areas outside of lane lines.
Also eventhough the model performs well on test samples, it can be in principal improved by adding augmented data and also experimenting with other SVM kernels.
Another option for improvment also is to smooth the boxes drawn from frame to frame by keeping a running average on the heatmaps.









