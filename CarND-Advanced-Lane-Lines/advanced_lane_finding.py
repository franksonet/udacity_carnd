import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from PIL import ImageFont, ImageDraw, Image
# plt.rcParams["figure.figsize"] = (8, 6)
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# from IPython.display import HTML
from time import time, sleep




# Show all output values
def bgr2rgb(bgr_img):
    '''
    Function to convert an BGR format image to RGB
    so that it can be presented proberply with plt.imshow()
    '''
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

def plot_one(img1, img1_title='', save_fname=None, bgr=True,
                 figsize=(12, 7), fontsize=25, cmap=''):
    '''
    Function to easily plot and save tow images for comparison
    '''
    if bgr == True and cmap =='':
        img1 = bgr2rgb(img1)
    if cmap == '':
        plt.imshow(img1);
    else:
        plt.imshow(img1, cmap=cmap)
    plt.title(img1_title, fontsize=fontsize);
    if save_fname != None:
        plt.savefig('output_images/{}.png'.format(save_fname));
    plt.show();
    plt.clf();

def plot_two(img1, img1_title, img2, img2_title,
                 save_fname=None, bgr=True,
                 figsize=(15, 6), fontsize=25):
    '''
    Function to easily plot and save tow images for comparison
    '''
    if bgr == True:
        img1 = bgr2rgb(img1)
        img2 = bgr2rgb(img2)
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize);
    f.tight_layout();
    ax1.imshow(img1);
    ax1.set_title(img1_title, fontsize=fontsize);
    ax2.imshow(img2);
    ax2.set_title(img2_title, fontsize=fontsize);
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.);
    if save_fname != None:
        plt.savefig('output_images/{}.png'.format(save_fname));

def calibrate(nx, ny, images, imgsize):
    '''
    Function for calibrating the camera
    '''
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for image in images:
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       imgsize, None, None)
    return mtx, dist

def undistort(img, mtx, dist):
    '''
    Function to undistort an image
    '''
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted

def color_threshold_s(img, s_thresh=(175, 255), sx_thresh=(30, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    abs_sobely = np.absolute(sobely)

    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary

def color_threshold_ls(img, s_thresh=(175, 255), l_thresh=(255, 255), sx_thresh=(30, 100)):
    print('using color threshold ls function...')
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    abs_sobely = np.absolute(sobely)

    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    ls_binary = np.zeros_like(s_channel)
    ls_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]) & (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, ls_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(ls_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary

def color_threshold_combined(img, s_thresh=(170, 255), sx_thresh=(20, 100)):

    def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=sx_thresh):
        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'y':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:  # if orient is x or anything else then y, set it to x
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobelx = np.absolute(sobelx)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return sxbinary


    def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

        # Return the binary image
        return binary_output


    def dir_threshold(img, sobel_kernel=3, thresh=(0.7, 1.3)):

        # Apply the following steps to img
        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        dir_grad = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(dir_grad)
        binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output


    # Run the function
    ksize = 3  # kernel size
    grad_x = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grad_y = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack((dir_binary, grad_x, mag_binary)) * 255
    combined = np.zeros_like(dir_binary)
    combined[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Define a function that applies Sobel x and y,
    # then computes the magnitude of the gradient
    # and applies a threshold
    img_s = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img_s, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    s_thresh = (220, 255)
    l_thresh = (170, 255)
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1]) & (l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    combined[(s_binary == 1) | (combined == 1)] = 1
    return color_binary, combined


def get_perspective_m(img, img_size, src, dst):
    '''
    Function to compute the perspective transform matrix
    '''
    # Compute the perspective transform
    M = cv2.getPerspectiveTransform(src, dst)

    # Inverse the warp
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def warp(img, M, img_size):
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

def initial_find_lane(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    #print('Shape of the out_img: {}'.format(out_img.shape))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting points for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[buffer:midpoint]) + buffer
    #print('leftx_base: {}'.format(leftx_base))
    rightx_base = np.argmax(histogram[midpoint:(imgsize[0]-buffer)]) + midpoint
    #print('rightx_base: {}'.format(rightx_base))

    # Choose the nuber of sliding windows
    nwindows = 9
    # Set the height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y position of all nonzero pixels in the image
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    # nonzero 's content: tuple(array of y of nonzero pixels, array of x of nonzero pixels)
    #print('Number of nonzero pixels: {}'.format(len(nonzero[0])))
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Setp through the windows one by one
    for window in range(nwindows):
        # Indentity window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - init_margin
        win_xleft_high = leftx_current + init_margin
        win_xright_low = rightx_current - init_margin
        win_xright_high = rightx_current + init_margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,
                      (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img,
                      (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        # good_lef_inds and good_right_inds contains the indexes of the pixels which falls in the conditions below
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found more than minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concateneate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(left_lane_inds) > minpix and len(right_lane_inds) > minpix:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    else:
        left_fit, right_fit, left_fitx, right_fitx = None, None, None, None
    return out_img, left_fit, right_fit, leftx, lefty, rightx, righty,left_fitx, right_fitx, histogram

def find_lane(binary_warped, left_fit, right_fit):
    # Identify the x and y position of all nonzero pixels in the image
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    # nonzero 's content: tuple(array of y of nonzero pixels, array of x of nonzero pixels)
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(left_lane_inds) > minpix and len(right_lane_inds) > minpix:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    else:
        left_fit, right_fit, left_fitx, right_fitx = None, None, None, None
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    return out_img, left_fit, right_fit, leftx, lefty, rightx, righty, left_fitx, right_fitx


def process_out_img(out_img, left_fit, right_fit,
                             left_fitx,right_fitx,
                             leftx, lefty, rightx, righty):

    # Visualize all the lane pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # # Draw two curves onto out_img using cv2.polylines()
    curves = np.zeros_like(out_img)
    pts_left = np.vstack((left_fitx, ploty)).T
    pts_left = np.int32([pts_left])
    cv2.polylines(curves, pts_left, isClosed=False, color=[255, 255, 0], thickness= 4)
    pts_right = np.vstack((right_fitx, ploty)).T
    pts_right = np.int32([pts_right])
    cv2.polylines(curves, pts_right, isClosed=False, color=[255, 255, 0], thickness= 4)

    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    out_img = cv2.addWeighted(out_img, 1, curves, 1, 0)
    return out_img

def evaluate(y_eval, ploty, left_fitx, right_fitx):
    '''
     Calculate the metrics for later use of sanity check
    '''
    # Fit new polynomials to x,y in world space
    ym_eval = y_eval * ym_per_pix
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * ym_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * ym_eval  + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    # If curverad is larger than 1500, then set it to 1500, cause 1500 or larger of curverad is close to straightline
    if left_curverad > 1500:
        left_curverad = 1500.0
    if right_curverad > 1500:
        right_curverad = 1500.0

    # Calculate the offset of the vehicle to the lane center
    x_left_bottom = left_fit_cr[0] * ym_eval**2 + left_fit_cr[1] * ym_eval + left_fit_cr[2]
    x_right_bottom = right_fit_cr[0] * ym_eval**2 + right_fit_cr[1] * ym_eval + right_fit_cr[2]
    offset = x_left_bottom + (x_right_bottom - x_left_bottom) // 2 -  imgsize[0]*xm_per_pix // 2

    # Calculate the delta between the maximum and minimum lanes distance
    lane_width = (right_fitx - left_fitx) * xm_per_pix
    min_max_lane_dist_delta = np.max(lane_width) - np.min(lane_width)

    # Calcutlate the lane width at y_eval
    # print(y_eval)
    lane_width_y_eval = lane_width[int(y_eval)]
    return (left_curverad, right_curverad), offset, lane_width_y_eval, min_max_lane_dist_delta

def sanity_check(curverad, offset, lane_width_y_eval, min_max_lane_dist_delta):
    '''
    Function to check if the found lanes are valid
    '''
    curverad_ok = abs(curverad[0] - curverad[1]) <= 800.
    lane_width_ok = lane_width_y_eval >= 3.1 and lane_width_y_eval <= 4.2
    lane_width_minmax_ok = min_max_lane_dist_delta <= 1.

    if curverad == None or offset == None or lane_width_y_eval == None or min_max_lane_dist_delta == None:
        detected = False
    elif curverad_ok and lane_width_ok and lane_width_minmax_ok:
        detected = True
    else:
        detected = False
    return detected

def warpback(warped, undist, left_fitx, right_fitx, ploty, Minv, img_size):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def add_text(img, text=''):
    img_new = np.copy(img)
    lines = str(text).split('\n')
    height = 0
    font=cv2.FONT_HERSHEY_SIMPLEX
    for line in lines:
        img_new = cv2.putText(img_new, line, (10,height),font,0.8,(255,255,255),2)
        height += 30
    return img_new


def add_text2(img, text='', fontsize=25, color=(255,255,255,255)):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img_new = Image.fromarray(img).convert('RGBA')
    rect_bg = Image.new('RGBA', img_new.size, (255,255,255,0))
    rect_draw = ImageDraw.Draw(rect_bg)
    rect_draw.rectangle([(5,10),(550,170)], fill=(0,0,0,128), outline=None)
    font1 = ImageFont.truetype('Roboto_Mono/RobotoMono-Bold.ttf', 15)
    rect_draw.text((1200, 690), '@frankso', (255,255,255,128), font=font1)
    img_new = Image.alpha_composite(img_new, rect_bg)
    draw = ImageDraw.Draw(img_new)
    # use a truetype font
    font2 = ImageFont.truetype('Roboto_Mono/RobotoMono-Bold.ttf', fontsize)
    lines = str(text).split('\n')
    height = 0
    for line in lines:
        if line == '' or line == None:
            line = ' '
        # Draw the text
        draw.text((25, height), line, color, font=font2)

        height += (fontsize + 5)

    img_new = img_new.convert('RGB')
    result = cv2.cvtColor(np.array(img_new), cv2.COLOR_RGB2BGR)
    return result

def pipeline(video_frame, debug=False, only_final=False):
    # Convert input image from RBG to BGR
    init_img = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
    if debug == True:
        plot_one(init_img, 'init_img')

    # Undistort the camera image
    undist = undistort(init_img, mtx, dist)
    if debug == True:
        plot_one(undist, 'undist')
    # Apply color threshold on S channel
    thresholded_colored, thresholded_binary = color_threshold_s(undist)
    if debug == True:
        thresholded_colored = np.uint8(thresholded_colored)
        plot_one(thresholded_colored, 'thresholded_colored')
        plot_one(thresholded_binary, 'thresholded_binary', cmap='gray')
    # Apply perspective transform
    birdview_binary = warp(thresholded_binary, M, imgsize)
    if debug == True:
        plot_one(birdview_binary, 'birdview_binary', cmap='gray')


    # Reset the lane finding after n iteration of failure or first time lane finding on first frame
    if True not in l_line.detected:
        print('Initialize the lane finding....')

        if debug == True:
            print('In initial lane finding...')

        init_lane_detction_binary, left_fit, right_fit,\
        leftx, lefty, rightx, righty, left_fitx, right_fitx,\
        histogram = initial_find_lane(birdview_binary)

        if debug == True:
            plt.plot(histogram)
            plt.show()
            plt.clf()

        detected = False

        if left_fit != None:
            init_lane_detction_binary = process_out_img(init_lane_detction_binary, left_fit, right_fit, left_fitx, right_fitx,leftx, lefty, rightx, righty)

            init_lane_detction_binary= np.uint8(init_lane_detction_binary)
            if debug == True:
                plot_one(init_lane_detction_binary, 'init_lane_detction_binary')

            y_eval = int(np.max(ploty))
            curvature, offset, lane_width, lane_dist_delta = evaluate(y_eval, ploty, left_fitx, right_fitx)

            if debug == True:
                print(curvature, offset, lane_width, lane_dist_delta)
            detected = sanity_check(curvature, offset, lane_width, lane_dist_delta)

        # detected = True
        if detected == True:

            if debug == True:
                print('Initial lane detected...')

            l_line.recent_xfitted = []
            r_line.recent_xfitted = []
            l_line.bestx = left_fitx
            r_line.bestx = right_fitx
            l_line.best_fit = left_fit
            r_line.best_fit = right_fit
            l_line.current_fit = np.array(left_fit)
            r_line.current_fit = np.array(right_fit)
            l_line.initial_lane_found = True

        else:
            # if initial lane not found, then leave the init_img no change
            l_line.initial_lane_found = False
            cv2.imwrite('lane_not_found_img/fail_{:.4f}.jpg'.format(time()), init_img)



    if l_line.initial_lane_found == True: # only execute in case the inital lane was found
        if debug == True:
            print('In normal lane finding...')

        lane_detction_binary, left_fit, right_fit, leftx, lefty, rightx, righty, left_fitx, right_fitx = find_lane(birdview_binary, l_line.best_fit, r_line.best_fit)
        lane_detction_binary= np.uint8(lane_detction_binary)

        detected = False

        if left_fit != None:
            lane_detction_binary = process_out_img(lane_detction_binary, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty)

            if debug == True:
                plot_one(lane_detction_binary, 'lane-detection-binary', cmap='gray')

            y_eval = np.max(ploty)
            curvature, offset, lane_width, lane_dist_delta = evaluate(y_eval, ploty, left_fitx, right_fitx)
            l_line.curverad_history.append(curvature)
            l_line.offset_history.append(offset)
            l_line.lane_width_history.append(lane_width)
            l_line.lane_width_delta_history.append(lane_dist_delta)

            if debug == True:
                print(curvature, offset, lane_width, lane_dist_delta)

            detected = sanity_check(curvature, offset, lane_width, lane_dist_delta)

        l_line.detected.append(detected)
        l_line.detected = l_line.detected[-n:]

        if detected == True:
            l_line.recent_xfitted.append(left_fitx)

            if debug == True:
                print(left_fitx.shape, 'left_fitx.shape')
            # Keep the last n iteration
            l_line.recent_xfitted = l_line.recent_xfitted[-n:]
            r_line.recent_xfitted.append(right_fitx)
            # Keep the last n iteration
            r_line.recent_xfitted = r_line.recent_xfitted[-n:]

            l_line.bestx = np.uint32(np.mean(l_line.recent_xfitted, axis=0))
            r_line.bestx = np.uint32(np.mean(r_line.recent_xfitted, axis=0))

            if debug == True:
                print(l_line.bestx.shape)
                print(ploty.shape)

            l_line.best_fit = np.polyfit(ploty, l_line.bestx, 2)
            r_line.best_fit = np.polyfit(ploty, r_line.bestx, 2)
            l_line.diffs = left_fit - l_line.best_fit
            r_line.diffs = right_fit - r_line.best_fit
            l_line.current_fit = np.array(left_fit)
            r_line.current_fit = np.array(right_fit)
            curvature, offset, lane_width, lane_dist_delta = evaluate(y_eval, ploty, l_line.bestx, r_line.bestx)

            l_line.radius_of_curvature = curvature[0]
            r_line.radius_of_curvature = curvature[1]
            l_line.offset = offset
            # r_line.line_base_pos = offset
            l_line.lane_width = lane_width
            l_line.lane_dist_delta = lane_dist_delta
            l_line.allx, r_line.allx = leftx, rightx
            l_line.ally, r_line.ally = lefty, righty

            # Set the format text for display on final output
            text = '\
                \nRadius of Curve:    {:.0f}m\
                \nOffset to Center:   {:.2f}m\
                \nDistance of Lanes:  {:.2f}m\
                \nDiff of Lane Dist.: {:.2f}m\
                '.format((l_line.radius_of_curvature+r_line.radius_of_curvature)/2,
                    l_line.offset, lane_width,lane_dist_delta)

        elif detected == False:
            # Set the format text for display on final output in case that lane not found
            # cv2.imwrite('lane_detection_binary.png', lane_detction_binary)
            text = '.\
                \nRadius of Curve:    {:.0f}m\
                \nOffset to Center:   {:.2f}m\
                \nDistance of Lanes:  {:.2f}m\
                \nDiff of Lane Dist.: {:.2f}m\
                '.format((l_line.radius_of_curvature+r_line.radius_of_curvature)/2,
                    l_line.offset, l_line.lane_width, l_line.lane_dist_delta)


        img_with_lane_boundary = warpback(birdview_binary, undist,
                                      l_line.bestx, r_line.bestx, ploty, Minv, imgsize)

        final_output = add_text2(img_with_lane_boundary, text)
    else:
        # final_output = undist
        text = '\
            \nERROR:\
            \nLane lines NOT detected in\
            \n last {} frames !!!\
            '.format(n)

        final_output = add_text2(undist, text, color=(255,0,0,255))


    if debug == True or only_final == True:

        f, axe = plt.subplots(3, 3, figsize=(10,10))
        f.tight_layout()
        axes = axe.flatten()
        axes[0].imshow(init_img)
        axes[1].imshow(undist)
        axes[2].imshow(thresholded_colored)
        axes[3].imshow(thresholded_binary, cmap='gray')
        axes[4].imshow(birdview_binary, cmap='gray')
        axes[5].plot(histogram)
        axes[6].imshow(init_lane_detction_binary, cmap='gray')
        axes[7].imshow(lane_detction_binary, cmap='gray')
        axes[8].imshow(final_output)

        plt.show()
        # cv2.imwrite('output_images/10_final_output.png', img)
    # Convert input image from RBG to BGR

    # cv2.imwrite('lane_not_found_img/fail_{:.4f}.jpg'.format(time()), init_img)
    return cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR)


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.initial_lane_found = False
        # was the line detected in the last iteration?
        self.detected = []
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0.0
        #distance in meters of vehicle center from the line
        self.offset = 0.0
        #lane width
        self.lane_width = 0.0
        self.lane_dist_delta = 0.0
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self.curverad_history = []
        self.offset_history = []
        self.lane_width_history = []
        self.lane_width_delta_history = []
        self.test = 'changed again'

def process_video(input_name, output_name, subclip=None):
    '''
    Function to process and output the input video
    '''
    if subclip == None:
        clip1 = VideoFileClip(input_name)
    else:
        clip1 = VideoFileClip(input_name).subclip(*subclip)
    output_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    output_clip.write_videofile(output_name, audio=False)


nx = 9  # the number of inside corners in x
ny = 6  # the number of inside corners in y
# Read in all calibration chessboard images
cal_images = glob.glob('camera_cal/calibration*.jpg')
imgsize = (1280, 720)
# Generate x and y values for plotting
ploty = np.linspace(0, imgsize[1] - 1, imgsize[1])

# Calibrate the camera
mtx, dist = calibrate(nx, ny, cal_images, imgsize)

# Four source coordinates
src_pts = np.float32(
    [[688, 450],
     [1120, 720],
     [192, 720],
     [593, 450]])

# four desired coordinates
buffer = 200  # The ammount of more pixels to contain in the birdview image
# to make sure the lanes are always contained no matter how big
# the curvature is.
dst_pts = np.float32(
    [[1120 - buffer, 0],
     [1120 - buffer, 720],
     [192 + buffer, 720],
     [192 + buffer, 0]])

# Use straight line image for computing the matrix of Perspective transform
straightline_img = cv2.imread('test_images/straight_lines1.jpg')
M, Minv = get_perspective_m(straightline_img, imgsize, src_pts, dst_pts)

# Set the width of the windows +/- margin
init_margin = 100
margin = 60
# Set minimum number of pixels found to recenter window
minpix = 50

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension as US regulation
standard_lane_width_in_px = (1120 - buffer) - (192 + buffer)
xm_per_pix = 3.7 / standard_lane_width_in_px  # meters per pixel in x dimension as US regulation

# Set n of iterations of data to store in the line calss
n = 5

# Create an instance of Line Class to sotre all the characteristics of each line detection
l_line = Line()  # left line
r_line = Line()  # right line


def main():

    input_video_name = 'project_video.mp4'
    output_video_name = 'output_project_video_{:.0f}.mp4'.format(time())
    process_video(input_video_name, output_video_name)

    print('Curverad history:')
    print('lr_min:', np.min(np.array(l_line.curverad_history)[:,0]))
    print('lr_max:', np.max(np.array(l_line.curverad_history)[:,0]))
    print('rr_min:', np.min(np.array(l_line.curverad_history)[:,1]))
    print('rr_max:', np.max(np.array(l_line.curverad_history)[:,1]))
    print('\noffset history:')
    print('offset min:', np.min(l_line.offset_history))
    print('offset max:', np.max(l_line.offset_history))
    print('\nlane width history:')
    print('lane_width min:', np.min(l_line.lane_width_history))
    print('lane_width max:', np.max(l_line.lane_width_history))
    print('\nlane delta history:')
    print('lane_delta min:', np.min(l_line.lane_width_delta_history))
    print('lane_delta max:', np.max(l_line.lane_width_delta_history))



    # input_video_name = 'harder_challenge_video.mp4'
    # output_video_name = 'output_harder_challenge_video_{:0f}.mp4'.format(time())
    # process_video(input_video_name, output_video_name, (0, 5))

def main_test():
    fname = 'lane_not_found_img/fail_1507806412.5450.jpg'
    init_img = mpimg.imread(fname)
    # plt.imshow(bgr2rgb(init_img));
    pipeline(init_img, debug=False, only_final=True);

if __name__ == '__main__':
    main()


