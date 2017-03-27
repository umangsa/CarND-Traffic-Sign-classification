#**Traffic Sign Recognition** #



**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./German_signs/30_end.jpg "End of speed limit 30 KMPH - a class the network never saw"
[image5]: ./German_signs/30_end_2.jpg "End of speed limit 30 KMPH - a class the network never saw"
[image6]: ./German_signs/bicycle_crossing.jpg "Bicycle Crossing"
[image7]: ./German_signs/bicycle_crossing_2.jpg "Bicycle Crossing"
[image8]: ./German_signs/keep_left.jpg "Keep Left"
[image9]: ./examples/before_preprocessing.png "Signs before and pre processing was done"
[image10]: ./examples/after_processing.png "Signs after improving contrast"
[image11]: ./examples/final_class_distribution.png "Image distribution after augmenting the training set"
[image12]: ./German_signs/keep_right.jpg "Keep Right"
[image13]: ./German_signs/no_entry.jpg "No Entry"
[image14]: ./German_signs/priority_road.jpg "Priority Road"
[image15]: ./German_signs/straight_or_right.jpg "Go Straight or Turn Right"
[image16]: ./examples/end_of_30.png "End of speed limit 30 KMPH"
[image17]: ./examples/end_of_30_2.png "End of speed limit 30 KMPH"
[image18]: ./examples/bicycle_crossing.png "Bicycle Crossing"
[image19]: ./examples/bicycle_crossing_2.png "Bicycle Crossing"
[image20]: ./examples/keep_left.png "Keep Left"
[image21]: ./examples/keep_right.png "Keep Right"
[image22]: ./examples/no_entry.png "No Entry"
[image23]: ./examples/priority_road.png "Priority Road"
[image24]: ./examples/go_straight_or_go_right.png "Go Straight or Turn Right"



###Data Set Summary & Exploration###

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 3rd code cell of the IPython notebook.  

Following are the basic statistics of the data provided for the project:
- Number of training examples = 34799
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th to 5th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. Following is the histogram of the training, validation and test data provided.


![alt text][image1]


From the historgram, it is evident that the data is not uniformly distributed across the classes. This could lead the network to get trained or overfit towards the classes that has much higher representation. We should be able to address this in the data augmentation side.

However, on the flip side, it also appears that the training, validation and test data are all having a similar distribution. So, for this set, we should not have to worry too much in this particular project. In an ideal world, we would still want to ensure a near equal distribution of data.


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for Pre processing is in cells 7 to 15. 

I found that there were many images that had very poor contrast as well as were very dark. I decided to improve the contrast using histogram equilization. I converted the images to grayscale and did the histogram equalization on the grayscale images. However, I found that this turned out to be extreme as I started losing information from several images as useful portions of the images started appearing burnt out. So I used Clahe transform to get better control of contrast improvement. 

After equalization, I converted the images back to RGB. So now I ended up having grayscale images in RGB. 

Following are the images that were available as part of the training set
![alt text][image9]

Here are the images after contrast improvement and converting back to RGB
![alt text][image10]

As a last step, I normalized the image data to -1 to + 1. This help to make training faster and avoid getting stuck in a local minima.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I did not split my data into training, validation and test data as the project already had provided this split in the data set. 

In cells 11-13, I applied several image transformation techniques to augument the data. When I looked at a sample image for each sign, I noticed that I cannot blindly apply every transformation to each class without changing the meaning of the sign. e.g. a vertical mirror of keep right will change the traffic sign to keep left. I could have done this and changed the name the traffic sign in order to get a larger data set. However, I did not do this in the project.

My final training set was augmented to 540,000 images. Initially I had 35,000 images in the training set. Following are the transformations I did to the images as part of augmentation:

- Vertical mirror : for the signs that would not change meaning
- Horizontal mirror: for the signs that would not change meaning
- Random rotation +- 10 degrees
- Added Gaussian Noise
- Added Speckle
- Combination of the above

I tried to not skew the distribution of the classes by applying conditional augmentation to signs that already had a lot of images. However, I was not very strict or rigid while trying to limit this aspect. 

Here is my final class distribution:

![alt text][image11]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 16th cell of the ipython notebook. I took the LeNet architecture and made changes to the architecture.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Dropout				|												|
|:---------------------:|:---------------------------------------------:| 
| Convolution 5x5     	| 2x2 stride, same padding, outputs 14x14x18 	|
| RELU					|												|
|:---------------------:|:---------------------------------------------:| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 					|
|:---------------------:|:---------------------------------------------:| 
| Flatten				| 												|
| Fully connected		| etc.        									|
| RELU					|												|
| Dropout				|												|
|:---------------------:|:---------------------------------------------:| 
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|:---------------------:|:---------------------------------------------:| 
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 21st cell of the ipython notebook. 

To train the model, I used an Adam Optimizer.

Following are the parameters I used during training:
Learning rate = 0.001
EPOCHS = 30
BATCH_SIZE = 128

I also added a L2 regularization and dropouts to avoid overfitting. The parameters for these are:
alpha = 0.001
Dropout = 0.75 (during training and 1 during test)


####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 19th and 20th cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 95.7% 
* test set accuracy of 95.3%

To arrive at the final solutions, I had to undergo several iterations. 

I started with choosing the LeNet architecture as is as it appeared to give a good accuracy without data augmentation or modifications. 

I faced a large number of problems:
* Initially I noticed that each time I reran the training, my validation accuracy would vary. The difference was a lot and meant that the model was not working very well.
* Enhanced the image contrast using Clahe
* Applied data augmentation. Initially I applied it uniformly across all images. This had a huge negative impact on the accuracy. To resolve this, I ensured that the image mirroring was applied to signs that would not change meaning.
* Tried to not skew the distribution of images across the classes
* Instead of using the B&W images, I converted them back to RGB. 
* Made the network wider by increasing the number of channels in each layer by a factor of 3
* Removed the Max Pool layer in LeNet model in Layer 1 and added a new layer instead. This layer was a convolution using a stride of 2
* Added L2 regularization of dropouts to reduce overfitting


I also tried tuning various parameters to get improve the accuracy.

Initially, I changed the batch size to 512, 1024 and larger numbers, increasing the number of iterations. I was training against a good GPU, hence thought this would help in improving speed of training. Also a larger batch meant the network saw larger number of images at once and would not overfit quickly. But during the course, I realized that this was not helping me much. Keeping a lower batch size gave me the best and most stable results. I think this is because the network was adjust weights faster and hence the results were better. Probably, for high variance image sets, a larger batch size would have more benefits.

Weights initialization. I was initializing the weights with a very lowe std dev = 0.0005. But this did not give me very good results. Finally I stuck to 0.01 as I saw best results using that.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image12] 
![alt text][image13] ![alt text][image14] ![alt text][image15]


The first and second image might be difficult to classify because this image was never shown to the network while training. The closest image that it saw was for end of speed limit 80 or end of all speed limit or restrictions. I expected the network would be able to classify this image in that category. However, it was surprising to see that it is classifying it as Keep Right

Next 2 images were for bicycle crossing. In 1 of the variants, it saw the arrows in the sign and classified it as a slippery road. I think adding more signs in training for this class and maybe more Epochs for training would help

The remaining images were easy to classify as the main features in these images were very clear and hence the network classified them correctly with high confidence.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 24th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| End of 30 speed limit | Keep right   									| 
| Bicycle crossing		| Slippery road or Bunpy road					|
| Keep Left				| Keep Left										|
| Keep Right			| Keep Right					 				|
| No Entry				| No Entry										|
| Priority Road			| Priority Road									|
| Go Straight or Right	| Go Straight or Right							|



The model was able to correctly guess 5 of the 9 traffic signs, which gives an accuracy of 50%. This shows that the network could have been trained better.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 25th - 27th  cell of the Ipython notebook.

Here are the predictions for the images

![alt text][image16] ![alt text][image17] ![alt text][image18] 
![alt text][image19] ![alt text][image20] ![alt text][image21] 
![alt text][image22] ![alt text][image23] ![alt text][image24]

