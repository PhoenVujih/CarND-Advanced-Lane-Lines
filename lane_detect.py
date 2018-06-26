import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from threshold_binary import *
from moviepy.editor import VideoFileClip

dist_pickle = pickle.load(open('cali_warp.p', 'rb'))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]
dst = dist_pickle["dst"]
src = dist_pickle["src"]
M = dist_pickle["M"]

def undist(img):
    """
    Undistort the image
    """
    return cv2.undistort(img, mtx, dist, None, mtx)

def warp(img):
    """
    Warp the image 
    """
    imsize = (img.shape[1], img.shape[0])
    img_udist = undist(img)
    return cv2.warpPerspective(img_udist, M, imsize)



window_width = 110 
window_height = 180 # Break image into 4 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_lane(image, window_width, window_height, margin):
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    for i in range(window_width):
        window[i] = -i*(i-window_width)
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
    
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(image)
    r_points = np.zeros_like(image)

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,image,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,image,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 1
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 1
        
    # Extract the two lanes seperately. 
    l_lane = l_points*image
    r_lane = r_points*image        
    
    return window_centroids, l_lane, r_lane


class Line():
    """
    Define the Line class
    """
    def __init__(self):
        self.img = None
        self.detected = False
        self.current_fit = [np.array([False])] 
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.com_fit = None  
        self.diffs = np.array([0,0,0], dtype='float') 
        
        self.ldetected = False
        self.lcurrent_fit = [np.array([False])] 
        self.lradius_of_curvature = None
        self.lline_base_pos = None
        self.rdetected = False
        self.rcurrent_fit = [np.array([False])] 
        self.rradius_of_curvature = None
        self.rline_base_pos = None
        
        self.lfit_list = []
        self.rfit_list = []
        
        self.l_bestfit = None
        self.r_bestfit = None
        
        self.radius = None
        self.offset = None
        
        self.limg = None
        self.rimg = None
        
    def add_fit(self, limg, rimg, mx=3.75/900, my=20/700):
        """
        Add the fitting parameters to the variable using the seperated lane binaries
        """
        self.limg = limg
        self.rimg = rimg
        # If No nonzero point is found in the two images then return None and False
        if len(np.nonzero(self.limg))==0 |  len(np.nonzero(self.rimg))==0:
            self.ldetected = False
            self.lcurrent_fit = [np.array([False])] 
            self.lradius_of_curvature = None
            self.lline_base_pos = None
            self.rdetected = False
            self.rcurrent_fit = [np.array([False])] 
            self.rradius_of_curvature = None
            self.rline_base_pos = None
        # Otherwise, Calculate the parameters. 
        else:
            self.limg = limg
            self.rimg = rimg
            self.ldetected = True
            # Extract the coordinations of the nonzero points in the image. Note that the horrizontal axis is x while the vertical is y. 
            [ly_points, lx_points] = np.nonzero(self.limg)
            # Fit the left lane using 2-order polynomial. Here I corrected the scale of the image before fitting. 
            self.lcurrent_fit = np.polyfit(ly_points*my, lx_points*mx, 2)
            # Calculate the y value of the base point
            self.ly_base = my*np.float(self.limg.shape[0])
            ly_base = my*self.limg.shape[0]
            # Calculate the radius of the curvature of the lane. Note that I didn't use 'np.abs' to get the absolute the value. 
            self.lradius_of_curvature = ((1 + (2*self.lcurrent_fit[0]*ly_base + self.lcurrent_fit[1])**2)**1.5) /(2*self.lcurrent_fit[0])
            # Calculate the horizontal location of the base point
            self.lline_base_pos = self.lcurrent_fit[0]*ly_base**2 + self.lcurrent_fit[1]*ly_base**1 + self.lcurrent_fit[2]*ly_base**0-mx*np.float(self.limg.shape[1])/2
            self.lfit = self.lcurrent_fit

            # Calculate the parameters of the right lane
            self.rdetected = True
            [ry_points, rx_points] = np.nonzero(self.rimg)
            self.rcurrent_fit = np.polyfit(ry_points*my, rx_points*mx, 2)
            self.ry_base = my*np.float(self.rimg.shape[0])
            ry_base = my*self.rimg.shape[0]
            self.rradius_of_curvature = ((1 + (2*self.rcurrent_fit[0]*ry_base + self.rcurrent_fit[1])**2)**1.5) /(2*self.rcurrent_fit[0])
            self.rline_base_pos = self.rcurrent_fit[0]*ry_base**2 + self.rcurrent_fit[1]*ry_base**1 + self.rcurrent_fit[2]*ry_base**0-mx*np.float(self.rimg.shape[1])/2
            self.rfit = self.rcurrent_fit
    
    def best_fit(self, mx=3.75/900, my=20/700):
        """
        Get the best fit of the lane and calculate the parameters.
        """
        # Check if the result of the current fit is reliable
        if 1/(self.rradius_of_curvature*self.rradius_of_curvature)>1/(-1000000):
            if np.abs(self.rradius_of_curvature-self.lradius_of_curvature)/(self.rradius_of_curvature+self.lradius_of_curvature)<0.3:
                # Append the reliable fitting parameters to the list
                self.lfit_list.append(self.lcurrent_fit)
                self.rfit_list.append(self.rcurrent_fit)
        if isinstance(self.lfit_list, list): 
            if len(self.lfit_list)>4:
                # Choose the last two iterations and calculate the average of them as the best fit. 
                self.lfit = np.average(np.array(self.lfit_list[-1:]), axis=0)
                self.rfit = np.average(np.array(self.rfit_list[-3:]), axis=0)
        # Calculate the parameters. 
        self.lradius_of_curvature = ((1 + (2*self.lfit[0]*self.ly_base + self.lfit[1])**2)**1.5) /(2*self.lfit[0])
        self.lline_base_pos = self.lfit[0]*self.ly_base**2 + self.lfit[1]*self.ly_base**1 + self.lfit[2]*self.ly_base**0-mx*np.float(self.limg.shape[1])/2
        self.rradius_of_curvature = ((1 + (2*self.rfit[0]*self.ry_base + self.rfit[1])**2)**1.5) /(2*self.rfit[0])
        self.rline_base_pos = self.rfit[0]*self.ry_base**2 + self.rfit[1]*self.ry_base**1 + self.rfit[2]*self.ry_base**0-mx*np.float(self.rimg.shape[1])/2
        self.radius = 2/(1/self.lradius_of_curvature + 1/self.rradius_of_curvature)
        self.offset = (self.rline_base_pos + self.lline_base_pos)/2 
        
    def lane_region(self, mx=3.75/900, my=20/700, src=src, dst=dst):
        """
        Draw the lane region based on the parameters of the best fit.
        """
        # Extract the potential region of the road based on the parameters of the best fit. 
        y_grid, x_grid = np.mgrid[0:self.limg.shape[0],0:self.limg.shape[1]]
        lx_at_y_grid = self.lfit[0]*(y_grid*my)**2+self.lfit[1]*(y_grid*my)+self.lfit[2]
        rx_at_y_grid = self.rfit[0]*(y_grid*my)**2+self.rfit[1]*(y_grid*my)+self.rfit[2]
        lhalflane = np.zeros_like(self.limg)
        rhalflane = np.zeros_like(self.limg)
        lhalflane[lx_at_y_grid<x_grid*mx] = 1
        rhalflane[rx_at_y_grid>=x_grid*mx] = 1
        # Get the lane region by combining the two binaries together
        lane = threshold(lhalflane)&threshold(rhalflane)
        # Unwarp the image
        M2 = cv2.getPerspectiveTransform(src, dst)
        imsize = (self.limg.shape[1], self.limg.shape[0])
        lane = cv2.warpPerspective(lane.data, M2, imsize)
        return lane

def process_img(img):
    """
    Define the function of image processing including searching the lane, calculate the parameters of the best fit and plot the lane region on the original image.
    """
    warp_img = warp(img)
    thresh = combined_threshold(warp_img)
    window_centroids, left_lane, right_lane = find_lane(thresh, window_width, window_height, margin)
    lane.add_fit(left_lane, right_lane)
    lane.best_fit()
    offset = lane.offset
    radius = np.abs(lane.radius)/1000
    img_plot = np.copy(img)
    img_plot[:,:,1][lane.lane_region()>0] = 250
    label_str1 = 'Radius of curvature: %.2f km' % radius
    result = cv2.putText(img_plot, label_str1, (30,40), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    label_str2 = 'Vehicle offset: %.1f m' % offset
    result = cv2.putText(img_plot, label_str2, (30,70), 0, 1, (0,0,0), 2, cv2.LINE_AA)
    return img_plot

# Initialize the lane
lane = Line()

video_output1 = 'project_video_output.mp4'
video_input1 = VideoFileClip('project_video.mp4')
processed_video = video_input1.fl_image(process_img)
print('processing the video')
processed_video.write_videofile(video_output1, audio=False)

