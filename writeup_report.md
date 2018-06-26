## Writeup Report


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/dis_udist_chess.png "Undistorted"
[image2]: ./output_images/dis_udist_sample.png "Road Transformed"
[image3]: ./output_images/binaries.png "Binary Example"
[image4]: ./output_images/warp_unwarp.png "Warp Example"
[image5]: ./output_images/warp_unwarp_sample.png "Warp Road sample"
[image6]: ./output_images/fit.png "Fit Visual"
[image7]: ./output_images/logic.png "Logic"
[image8]: ./output_images/output_sample.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `calibration.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

I first stored the calibrated parameters and then read it when I needed it. The function of distortion correction is in `lane_detect.py`, line 17.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in `threshold_binary.py`). To make the logical operation of the binary images more convenient, I also define the class `threshold` and it's operation rules (at line 5 in `threshold_binary.py`)

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

The combined threshold was calculated using the following formula: `[(sobel_x & sobel_y) or S_Channel or yellow or white] not black` which was contained from line 129 to 142 in `threshold_binary.py`

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Actually, I firstly warped the image and then calculated the combined threshold of the image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 23 through 29 in the file `lane_detect.py` .  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points which were stored first for later use.  I chose the hardcode the source and destination points by picking out the specific points in the straight lane image:

![alt text][image4]

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 575, 465      | 245, 0        |
| 710, 465      | 1059, 0      |
| 1059, 702     | 1059, 720      |
| 245, 460      | 245, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I applied the method of sliding window search provided in the lesson to search for the lane and extract the nonzero points of each lane to fit with a 2nd polynomial like this:

![alt text][image6]

Here I used four windows to cover the lane. It is because that I tried out different numbers of windows and found that 4 windows is enough unless the curvature of the road is very large (small radius).

The code of lane finding can be found from line 42 to 94 in `lane_detect.py`


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To further get a more smooth accurate curve of the lane, I built a class named `Line` in `lane_detect.py` (line 97). The class contains the function of fitting, data storing and finding out the best parameters. The following figure shows how I get the best fitting parameters for later curvature and offset calculation (funtion `Line.best_fit`):

![alt text][image7]

Since I store the reliable data only, it can help increase the accuracy of the fitting curve. Moreover, since I use the average value of the last two reliable fitting parameters, the curve could be smoothened.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 218 through 235 in my code in `lane_detect.py` in the function `process_img()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Even though I use a combined threshold, there is a shortcoming in my algorithm: if the aperture or ISO of the camera is too large/high in a bright environment, most of the region in the image will become white and it will be regarded as the lane line. That's the main reason why my method doesn't work very well in the challenge and harder challenge video.

A possible solution is to further combine the edge detection technologies which may help abandon the large uniform region  filled with the same color (white). However, when the light is too strong, the image captured can hardly reflect the difference between the line and road surface. In this situation, intuitively, a time serial processing method might need to be employed to consider the historical lane location and predict the current lane parameters.

Another method we may employ is that define a standard to judge which part of lane line detected is reliable and which part is not. And then calculate the curve using the historical information and the reliable part.

Another problem may come from the small radius of curvature of the road. Small radius means large curvature, which may cause the miss of one side of the lane as shown in harder_challenge video. Time serial processing method may be a solution to predict the current location of the lane. Or we may use deep learning method like recurrent neural network and fully neural network to find out the lane in pixel level although it needs more calculation.
