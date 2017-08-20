# **Vehicle Detection Project**
## *by Maxim Taralov*
---
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_not_car.png
[image2]: ./images/features.png
[all_boxes]: ./images/all_boxes.png
[final_result1]: ./images/final1.jpg
[final_result2]: ./images/final2.jpg
[debug_frame1]: ./images/one_box_detection.jpg
[debug_frame2]: ./images/no_boxes_detection.jpg
[debug_frame3]: ./images/false_positives_filter.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #175 through #203 of the file called `pipeline.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

As there were many series of sequential images, during reading of the file names I selected the first 80% of each folder to go to training, and the rest for testing. This way, the nearly identical images from a single time series were restricted to either the train or the test set and there was less chance of overfitting on the whole data set during training. The reading of the image names and splitting them into train and test sets is in the function `read_images` in lines #7 to #33 in `helper_functions.py`.

The final size of the train set is `14206` images and the test set consists of `3554` images.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and finally settled on using `orientations=10`, `pixels_per_cell=(16, 16)` and `cells_per_block=(3, 3)` and the `YCrCb` color space. I use the features extracted from all the channels of the image laid out in a single feature vector. I used these parameters as they gave me the highest accuracy for the SVM classifier.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using only the HOG features described in the previous section. I didn't use any color information directly as using color features (either through histograms or through colors directly) hindered the training of the classifier. Using only HOG features, the linear SVM (class `sklearn.svm.LinearSVM`) with `C=0.5` finished training in only `0.73`s after `165` iterations and with `969` support vectors. In contrast, when using also color features, the training didn't manage to finish even in the 1000 iterations that are set by default in the method. The accuracy of the learned model was `99.73%` on the training set and `99.41%` on the test set, i.e. using the given HOG features, the dataset was almost linearly separable. I managed to increase the accuracy further to `99.6%` on the test by using hog features also from the `CIELUV` color space, but this is disabled in the final version of the code as it leads to slower computation during test time (although it can be enabled by setting `use_two_colors` to `True` in lines `#64` in `helper_functions.py` and `#36` in `pipeline.py`). Before passing the training set to the `svm.fit` function, the data was shuffled using `sklearn.utils.shuffle` and scaled using `sklearn.preprocessing.StandardScaler`. SVM training is in lines `#212 -- #220` in file `pipeline.py`.

---
### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window in lines `#32--#107` in `pipeline.py` in function `get_boxes`. I pass the functions which y-ranges I want to search, as well as the sizes of the sliding windows for the respective ranges. For each range given I extract it from the image, scale it with the ratio of the current sliding window size to `(64,64)`, which was the size of the images I used for training. I then compute the HOG only once per scale and then extract the features for each sliding window. After extraction, for each window I make a prediction using the SVM model and store it. In the end, I used three window sizes `(72,72), (96,96), (128,128)` for y in ranges `[400,560],[400,656],[432,656]`, respectively. I chose these sizes and ranges, as they gave me best coverage of possible car sizes. Since I use 16 pixels per cell for the HOG, I have only four cells in each direction per (64,64) image patch. As such, I can only have overlaps of 0%, 25%, 50% or 75%. I used 75% overlap. Here is an example of all the sliding windows drawn onto an image

![alt text][all_boxes]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features. I optimized the classifier by tuning the features as discussed previously. Here are some example images:

![alt text][final_result1]

![alt text][final_result2]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./video_output.mp4) . And here is a [link to a debug version](./video_output_debug.mp4) that shows the final result as well as the box detections and heatmaps.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

From the positive detections in each frame I created a heatmap. I kept the heatmaps from the last 15 frames processed. I then looked for cars based on two criteria. First, if in the last two frames the mean detection for some pixel is above 5 (i.e. this place was on average in 5 detected boxes in the last two frames), I consider it a strong detection and add it to a final heatmap for the current frame. The second criterion is based on number of detections. If a pixel was detected in at least 11 frames from the last 15 frames (regardless of the number of detections for each frame), I also add it to the final heatmap. This allows for detecting vehicles that are further away and thus not moving a lot from frame to frame, but are detected from only a small number of boxes. Note that in a 25Hz video, 15 frames are about half a second.

After assembling the final heatmap for the current frame, I use `scipy.ndimage.measurements.label()` to identify contiguous regions in the heatmap. For each such region, I constructed a bounding box, that I assumed to contain a car. The filtering and the car identification are implemented in lines `#117 -- #133` of `pipeline.py`.

Here are a few example frames from the debugging mode of the pipeline (accessed by passing `-d` when calling `pipeline.py`). In the top left corner, I show the image with all the possible bounding boxes in green, as well as the detected bounding boxes in blue. In the top right corner is the final result from the pipeline. In the bottom left corner is the heatmap that shows the mean of the last two frames plus the result of the second criterion described above (which is always a value of one or zero). In the bottom right corner is the final heatmap, after thresholding the mean, that I pass to `scipy.ndimage.measurements.label()`. For illustration all detections are set to max, since `label()` does not care, how strong a detection is.

In the first frame I show how even a single detection can result in a hit if it is a part of a long running series of detection.

![alt text][debug_frame1]

Actually, even frames without a single box detected, can result in a detected car.

![alt text][debug_frame2]

In the last example, I show that even frames that have relatively many false positives, can be effectively filtered

![alt text][debug_frame3]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem I faced, was that I tried to use both HOG features and color information for training the SVM. As already mentioned in a previous section, the method couldn't converge and in the end was giving only about 95% accuracy on the test set. This was insufficient for robust detection and there were a lot of false positives. When I moved to HOG features only, the detection went above 99%.

The second problem was devising a strategy for rejecting false positives. I tried to collect the positive detections from several consecutive frames and use them to construct the heatmap, but thresholding on that map was either rejecting some valid but weakly detected positions of a vehicle or allowing some false positives.

The last major problem was more of a speed issue. When using a naive sliding window approach, i.e. first select a window and then compute the features for it, the processing was running at less than 0.5 frames per second. When I moved to a single calculation of the HOGs and then simply extracted the features for each window, the processing time decreased by about an order of magnitude and the pipeline was running at 3FPS.

The detection can fail due to misprediction related either to not detecting some vehicle that is not present in the training set (e.g., some specialized agricultural or construction vehicles) or due to the fixed sizes of the sliding windows not accommodating a specific vehicle that can be too large due to its size or position (e.g., a car right in front of the hood might need bigger sliding windows). The first problem can be solved with more and different training images. The second can be solved by using either more sliding window sizes or alternative methods that don't use sliding windows at all, like [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) or [YOLO: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640). The thresholding strategies can fail due to an unfortunate run of mispredicted images. This can be mitigated by tracking individual cars and restricting the positions that a new car can emerge from (i.e., no car should just pop out right in front of us). Furthermore, my current pipeline runs at about 3 frames per second, which is potentially too slow for real-time detection. This can be improved by, e.g., using the SSD method mentioned before, where the authors claim 19-46 FPS.
