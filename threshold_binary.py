import numpy as np
import cv2
import matplotlib.image as mpimg

class threshold: 
    """
    Define the class of threshold and its calculation rules
    """
    def __init__(self, value):
        self.data = value
    def __and__(self, other):
        binary_output = np.zeros_like(self.data)
        binary_output[(self.data==1) & (other.data==1)]=1  
        return threshold(binary_output)
    def __or__(self, other):
        binary_output = np.zeros_like(self.data)
        binary_output[(self.data==1) | (other.data==1)]=1  
        return threshold(binary_output)
    def no(self, other):
        binary_output = np.zeros_like(self.data)
        binary_output[(self.data==1) & (other.data==0)]=1  
        return threshold(binary_output)
    
def img_channel(img, colorspace, channel_num):
    """
    Exxtract the data of single channel from the specific color space
    """
    if colorspace == 'RGB':
        img_channel = img[:,:,channel_num]
        return img_channel
    elif colorspace == 'HSV':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_channel = img[:,:,channel_num]
        return img_channel
    elif colorspace == 'HLS':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img_channel = img[:,:,channel_num]
        return img_channel
    else: 
        raise('Please check the inputs (img, colormap, channel number)')

def thresh(img, thresh=[0, 255]):
    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary_output

def mag_thresh(img,orient='x', thresh=(5,100)):
    """
    Threshold of magnitude of single sobel (x or y)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return threshold(binary_output)

def scale_sobel_thresh(img, sobel_kernel=3, mag_thresh=(5, 15)):
    """
    Threshold of magnitude of sobel
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return threshold(binary_output)

def dir_threshold(img, sobel_kernel=3, thresh=(np.pi/8, np.pi/2)):
    """
    Threshold of the direction of sobel
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return threshold(binary_output)

def hls_select(img, thresh=(170, 255)):
    """
    Select the S channel of HLS colorspace to define the threshold
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return threshold(binary_output)

def yellow_thresh(img):
    """
    Extract the yellow part based on RGB color space
    """
    thresh_R = threshold(thresh(img_channel(img, 'RGB', 0), thresh=[110, 255]))
    thresh_G = threshold(thresh(img_channel(img, 'RGB', 1), thresh=[140, 230]))
    thresh_B = threshold(thresh(img_channel(img, 'RGB', 2), thresh=[0, 110]))
    return thresh_R & thresh_G & thresh_B

def white_thresh(img):
    """
    Extract the white part based on RGB color space
    """
    thresh_R = threshold(thresh(img_channel(img, 'RGB', 0), thresh=[220, 255]))
    thresh_G = threshold(thresh(img_channel(img, 'RGB', 1), thresh=[220, 255]))
    thresh_B = threshold(thresh(img_channel(img, 'RGB', 2), thresh=[220, 255]))
    return thresh_R & thresh_G & thresh_B

def black_thresh(img):
    """
    Extract the black part based on RGB color space
    """
    thresh_R = threshold(thresh(img_channel(img, 'RGB', 0), thresh=[0, 40]))
    thresh_G = threshold(thresh(img_channel(img, 'RGB', 1), thresh=[0, 40]))
    thresh_B = threshold(thresh(img_channel(img, 'RGB', 2), thresh=[0, 40]))
    return thresh_R & thresh_G & thresh_B

def combined_threshold(img):
    """
    Define the combined threshold
    """
    sobel_x = mag_thresh(img, thresh=(30,100))
    sobel_scale = scale_sobel_thresh(img, mag_thresh=(20, 255))
    sobel_y = mag_thresh(img, orient='y', thresh=(40,255))
    hls_2 = hls_select(img, thresh=(140, 255))
    yellow = yellow_thresh(img)
    white = white_thresh(img)
    black = black_thresh(img)   
    binary_output = ((sobel_x&sobel_y)|white|yellow|hls_2).no(black)
    # Return the numpy array for the next step
    return binary_output.data

