# import libraries
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt
import cv2
import os, glob
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
import sys
import math

# this function creates the training data for class c for frame t.
# k = number of previous frames (default set arbitrarily to 10)
# list_of_RGB_Images = input images
# imagesMasks = correspond to either yellow lane vs. road or white lane vs. road
def createTrainingData(list_of_RGB_Images, imageMasks_w, imageMasks_y):
    # Create a list of all RGB values
    R = []
    G = []
    B = []
    for img in list_of_RGB_Images:
        # reshape each color val into an array
        currR = img[:,:,0].reshape(np.shape(img[:,:,0])[0]*np.shape(img[:,:,0])[1])
        currG = img[:,:,1].reshape(np.shape(img[:,:,1])[0]*np.shape(img[:,:,1])[1])
        currB = img[:,:,2].reshape(np.shape(img[:,:,2])[0]*np.shape(img[:,:,2])[1])

        # concatenate with R, G, and B
        R = np.concatenate((R, currR))
        G = np.concatenate((G, currG))
        B = np.concatenate((B, currB))

    y_w = []
    for mask in imageMasks_w:
        # for both white/yellow, there should be a high R value, so just take
        # red value, reshape, then set to binary
        currY = mask[:,:,0].reshape(np.shape(mask)[0]*np.shape(mask)[1])

        # convert currY to binary (0 = road, 1 = lane)
        currY = np.where(currY > 0, 1, 0)

        y_w = np.concatenate((y_w, currY))

    y_y = []
    for mask in imageMasks_y:
        # for both white/yellow, there should be a high R value, so just take
        # red value, reshape, then set to binary
        currY = mask[:,:,0].reshape(np.shape(mask)[0]*np.shape(mask)[1])

        # convert currY to binary (0 = road, 1 = lane)
        currY = np.where(currY > 0, 1, 0)

        y_y = np.concatenate((y_y, currY))

    # create training data to return
    X = np.zeros((len(R), 3))
    X[:,0] = R
    X[:,1] = G
    X[:,2] = B

    return X, y_w, y_y

# this function fits data using LDA to find conversion vector from previous images (X,y)
# then applies learned weights to current image (currImage)
def applyLDA(X, y):
    # create classifier object
    clf = LinearDiscriminantAnalysis()

    # fit classifier to data
    clf.fit(X, y)

    w = clf.coef_

    # scale between .1 and 1
    # import sklearn
    # scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(.1,1))
    # scaler = scaler.fit(w.reshape(-1,1))
    # w_scaled = scaler.transform(w.reshape(-1,1))
    #
    # w_scaled = w_scaled / np.linalg.norm(w_scaled,1)
    # w = np.transpose(w_scaled)

    w = np.abs(w)

    return w

# this function converts an image with the calculated gradient-enhancing vector
def convertToGray(w, img):
    grayImg = np.dot(img, w)

    #scale between 0 and 255
    img_vals = grayImg.flatten()
    grayImg = ((grayImg - min(img_vals)) / (max(img_vals) - min(img_vals))) * 255

    #grayImg = (w[0]*img[:,:,0] + w[1]*img[:,:,1] + w[2]*img[:,:,2]).astype(np.uint8)
    return np.uint8(grayImg)

# this function reads in the initial images and their respective masks
def readInitImages(crop_pct=.4):
    rgbImages = []
    masks_w = []
    masks_y = []
    imgNum = np.arange(1,6)
    for count in imgNum:
        currImg = cv2.imread('laneData/img'+str(count)+'.jpg')
        whiteMask = cv2.imread('laneData/img'+str(count)+'_lane_w.jpg')
        yellowMask = cv2.imread('laneData/img'+str(count)+'_lane_y.jpg')

        # crop y axis
        currImg = currImg[int(480*crop_pct):np.shape(currImg)[0],:]
        whiteMask = whiteMask[int(480*crop_pct):np.shape(whiteMask)[0],:]
        yellowMask = yellowMask[int(480*crop_pct):np.shape(yellowMask)[0],:]

        # add to list to return
        rgbImages.append(currImg)
        masks_w.append(whiteMask)
        masks_y.append(yellowMask)

    return rgbImages, masks_w, masks_y

def cov(a,b):
    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum / (len(a) - 1.0)

def gaussian_intersection_solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

def get_slope(line):
    x1, y1, x2, y2 = line[0]
    slope = ((y2-y1) / float(x2-x1))
    return slope

def filter_lines(lines,img_shape,thresh_h_percentage,slope_cutoff):
    """Lines should go through both the near and far region of an image. The cutoff for near and far is determined by thresh_h.
    Also, lane slopes should be >= slope_cutoff (should probably be around .3 since they should approach vertical lines"""
    final_lines = []
    h,w,depth = img_shape
    thresh_h = h * thresh_h_percentage
    for line in lines:
        for x1, y1, x2, y2 in line:
            if (y1 < thresh_h and y2 > thresh_h) or (y1 > thresh_h and y2 < thresh_h):  # far near
                slope = get_slope(line)
                if abs(slope) >= slope_cutoff: #slope is likely to approach "vertical" slopes
                    final_lines.append(line)
    return final_lines

def keep_part_of_image(img,h_percentage_to_keep):
    """only keep the bottom portion of the image to feed to hough. Zero out the rest"""
    mask = np.zeros_like(img)
    h, w = img.shape[:2]
    h_keep = int(h * (1-h_percentage_to_keep))

    # keep some bottom percentage of image
    mask[h_keep:h][0:w] = 1

    masked_img = cv2.bitwise_and(img,mask)
    return masked_img

def draw_lines(img,lines):
    """returns a copy of the img with the lines drawn on it"""
    line_img = img.copy()
    try:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    except:
        pass
    return line_img

def get_lowest_line(lines):
    """Note: Takes into account the slope of the line too. Prefers higher slopes. Also more extreme x values"""
    #actually the "lowest" line will be the one with the highest y
    max_loc = None
    max_value = -sys.maxsize
    for i in np.arange(0,len(lines)):
        line = lines[i]
        x1,y1,x2,y2 = line[0]
        y1 = y1 + 30*abs(get_slope(line))
        y2 = y2 + 30*abs(get_slope(line))
        if y1 > max_value:
            max_value = y1
            max_loc = i
        if y2 > max_value:
            max_value = y2
            max_loc = i
    return [max_loc,max_value]


def select_predictions(lines,img_w):
    """Given several lines left, make the best guess as to which are the final 2"""
    final_lines = []

    line_slope_tuples = list(map(lambda x: (x,get_slope(x)), lines))
    pos_slope_tuples = list(filter(lambda x: x[1] > 0, line_slope_tuples))
    neg_slope_tuples = list(filter(lambda x: x[1] <= 0, line_slope_tuples))

    if len(lines) == 0: #no predictions in this case...:(
        return final_lines
    if len(neg_slope_tuples) == 1 and len(pos_slope_tuples) == 1: #best case. 1 for each
        return lines
    elif len(neg_slope_tuples) == 1 or len(pos_slope_tuples) == 1:
        if len(neg_slope_tuples) == 1:
            final_line = neg_slope_tuples[0][0]
            final_lines.append(neg_slope_tuples[0][0])
            lines = list(map(lambda x: x[0],pos_slope_tuples))
        if len(pos_slope_tuples) == 1:
            final_line = pos_slope_tuples[0][0]
            final_lines.append(pos_slope_tuples[0][0])
            lines = list(map(lambda x: x[0],neg_slope_tuples))

        #choose the lowest line for the other side, if there are some to pick from
        if len(lines) > 0:
            i, min_val = get_lowest_line(lines)
            other_line = lines[i]

            final_lines.append(other_line)

    elif len(neg_slope_tuples) == 0 or len(pos_slope_tuples) == 0: #not getting one of the lanes...for other one, just return the one with lowest x
        min_i, min_val = get_lowest_line(lines)
        final_lines.append(lines[min_i])
    else:
        #both have more than 1

        #return the two lowest lines in each group
        pos_lines = list(map(lambda x: x[0], pos_slope_tuples))
        neg_lines = list(map(lambda x: x[0], neg_slope_tuples))

        if len(pos_lines) > 0:
            i,min_val = get_lowest_line(pos_lines)
            final_lines.append(pos_lines[i])
        if len(neg_lines) > 0:
            j,min_val = get_lowest_line(neg_lines)
            final_lines.append(neg_lines[j])

    return final_lines

def region_of_interest(lines,canny_img,width=20):
    if len(lines) > 2:
        raise Exception("Shouldn't be calculating a region of interest for len(lines) > 2")

    #sort lines by midpoint x value so always comes out in same order
    if len(lines) == 2:
        line1 = lines[0]
        line2 = lines[1]
        if ((line1[0][0]+line1[0][2])/2) > ((line2[0][0]+line2[0][2])/2):
            lines[0], lines[1] = lines[1], lines[0]

    masks = []
    for line in lines:
        mask = np.zeros_like(canny_img)

        line_points = line[0]

        left = [pt-(width/2) if i%2==0 else pt for i, pt in enumerate(line_points)]
        right = [pt + (width / 2) if i % 2 == 0 else pt for i, pt in enumerate(line_points)]

        poly = [np.array([ [left[0],left[1]], [left[2],left[3]],[right[2], right[3]],[right[0], right[1]] ],dtype=np.int32)]

        cv2.fillPoly(mask, poly,1)

        #only allowed to be up to 30% of image length
        pct = .2
        h,w = canny_img.shape
        bottom = max(left[1],left[3])
        limit = max(int(h * (1-pct)) - (h-bottom),int(h * (1-.35))) #still can't go past 35% of image though
        mask[0:limit][0:w] = 0

        display_img(mask)

        masks.append(mask)

    return masks

def display_img(im,cmap="gray",convertToRGB=False,show_img=False): #so I have a place to easily turn plotting on and off
    if show_img and convertToRGB:
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    if show_img:
        if cmap != "":
            plt.imshow(im,cmap=cmap) #need to set to gray because cv2 and plt do GBR vs RGB
        else:
            plt.imshow(im)
        plt.show(show_img)

# read in initial 5 images with respective masks
crop_pct = .4
list_of_RGB_Images, imageMasks_w, imageMasks_y = readInitImages(crop_pct)

# create training data
X, y_w, y_y = createTrainingData(list_of_RGB_Images, imageMasks_w, imageMasks_y)

img_number = 5
lane_means = [90,90] #save means used so can use again if algorithm gets off track
road_means = [50,50]
last_pos_line = None #save last lines predicted
last_neg_line = None
last_fs = [None,None] #save curve functions
last_xrange = [[None,None],[None,None]] #save info from curve fitting
used_last_frame_prediction = False
num_times_last_frame_prediction = 0
while img_number < 5619:
    img_number += 1

    print("\n*** Running on image " + str(img_number) + " ***")

    if img_number > 2500:
        break

    # for each mask, compute LDA
    colorMask = [y_w, y_y]
    gradientEnhancedVectors = []
    for mask in colorMask:
        weight = applyLDA(X, mask)
        gradientEnhancedVectors.append(weight)

    # read in next image
    img = cv2.imread('laneData/img' + str(img_number) + '.jpg')

     # test on next image and save result
    colorConv = ['w','y']
    count = 0
    canny_img = None
    for i in np.arange(0,2):
        w = gradientEnhancedVectors[i][0]

        if (w == 0).all(): #if have no predictions for a certain mask, will get all 0 weights. In that case, default to this
            w = np.array([.1,.4,.5])

        # convert image to gray with weight for respective class (w, then y)
        grayImg = convertToGray(w, img)

        # save image
        #cv2.imwrite("laneData/img" + str(i) + "_gray_" + str(colorConv[count]) + ".jpg", grayImg)

        # display image
        display_img(grayImg)

        count = count + 1

        #### Adaptive Canny Edge ###

        #look at last gray scale image, and get values for each mask
        mask = colorMask[i]
        last_img_mask = mask[-int(grayImg.shape[0]*grayImg.shape[1]*crop_pct):]
        last_gray_img = convertToGray(w,list_of_RGB_Images[-1])
        last_gray_img = last_gray_img.flatten()[-int(grayImg.shape[0] * grayImg.shape[1] * crop_pct):]

        lane_indices = np.nonzero(last_img_mask)
        lane_values = last_gray_img[lane_indices]
        road_indices = np.where(last_img_mask == 0)[0]
        road_values = last_gray_img[road_indices]

        if len(lane_values) == 0: #use the thresholds from last time
            print("No lane points for this mask, so using last means")
            lane_mean = lane_means[i]
            road_mean = road_means[i]
        else:
            lane_mean = np.mean(lane_values)
            road_mean = np.mean(road_values)

        lane_means[i] = lane_mean
        road_means[i] = road_mean

        print("Lane mean: " + str(lane_mean))
        print("Road mean: " + str(road_mean))

        th_small = road_mean
        th_large = lane_mean

        #print("Large threshold: " + str(th_large))
        #print("Small threshold: " + str(th_small))

        sub_canny_img = cv2.Canny(grayImg,th_small,th_large)
        display_img(sub_canny_img)

        if canny_img is None:
            canny_img = sub_canny_img
        else:
            canny_img = canny_img | sub_canny_img

    ### Keep only bottom part of canny image ###

    display_img(canny_img)

    canny_img = keep_part_of_image(canny_img,.5)
    display_img(canny_img)

    ### Hough Transform on canny image ###
    #TODO: May need to modify these more

    rho = 2 #distance resolution in pixels
    theta = np.pi/180 #angle resolution of accumulator in radians
    threshold = 110
    minimum_line_length = 80 #a line has to be at least this long
    maximum_line_gap = 250 #maximum allowed gap between line segments to treat them as a single line
    #Based on Robust Detection of Lines Using the Progressive Probabilistic Hough Transform by Matas, J. and Galambos, C. and Kittler, J.V.
    lines = cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minimum_line_length, maximum_line_gap)

    ### Filter the resulting lines ###

    thresh_h_percentage = .7
    slope_cutoff = .35
    lines = filter_lines(lines,img.shape,thresh_h_percentage,slope_cutoff)
    print("Num final lines: " + str(len(lines)))

    #draw lines on image to see results
    line_img = draw_lines(img,lines)
    display_img(line_img, cmap="", convertToRGB=True)

    #reduce down to just 2 lines with some heuristics. In future, would be better to do heuristic of choosing lanes that best overlap with labels from last image
    lines = select_predictions(lines,img.shape[1])
    print("Reduced to " + str(len(lines)))

    if len(lines) < 2:
        if num_times_last_frame_prediction < 8:
            print("Unable to make 2 predictions, so using help from last frame's predictions")
            if len(lines) == 0:
                lines.extend(last_neg_line)
                lines.extend(last_pos_line)
            if len(lines) == 1:
                pos_line = list(filter(lambda line: get_slope(line) > 0,lines))
                if len(pos_line) > 0:
                    lines.extend(last_neg_line)
                else:
                    lines.extend(last_pos_line)
            used_last_frame_prediction = True
            num_times_last_frame_prediction += 1
        else:
            pass #live with just one or no prediction
    else:
        used_last_frame_prediction = False
        num_times_last_frame_prediction = 0
    last_pos_line = list(filter(lambda line: get_slope(line) > 0,lines))
    last_neg_line = list(filter(lambda line: get_slope(line) < 0,lines))

    #draw lines on image to see results
    line_img = draw_lines(img,lines)
    display_img(line_img, cmap="", convertToRGB=True)

    #cv2.imwrite("predicted_lanes/img" + str(img_number).zfill(4) + "_ht.jpg", line_img)

    ## Region of Interest ## keep a region of interest around each line for lane edges to fit
    width = 55
    region_masks = region_of_interest(lines, canny_img, width)

    ## Curve Fitting ##
    curves_img = img.copy()
    lane_masks = [np.zeros_like(canny_img),np.zeros_like(canny_img)] #for future training steps
    for i in range(len(region_masks)):
        m = region_masks[i]
        #apply mask
        sub_canny_img = cv2.bitwise_and(canny_img,m)

        #plt.imsave("predicted_lanes/img" + str(img_number).zfill(4) + "_ksubcanny_" + str(i) + ".jpg",sub_canny_img)

        y_pts, x_pts = np.nonzero(sub_canny_img)
        if len(x_pts) < 100 and len(y_pts) < 100:
            print("Empty sub canny image (or few points), so using last curve function")
            f = last_fs[i]
            minx = last_xrange[i][0]
            maxx = last_xrange[i][1]
        else:
            f = np.poly1d(np.polyfit(x_pts,y_pts,2)) #curve function
            minx = min(x_pts)
            maxx = max(x_pts)
        last_fs[i] = f #save for later
        last_xrange[i][0] = minx
        last_xrange[i][1] = maxx

        x = np.arange(minx,maxx,.05)
        y = f(x)

        cpts = [np.array(a,dtype=np.int32) for a in zip(x,y)]

        red_color = [28,39,255] #cv2 is BGR
        cv2.polylines(curves_img, [np.array(cpts)],False,red_color,thickness=5)

        #lane masks for future training steps
        blank = np.zeros_like(sub_canny_img)
        mark_color = [255,255,255]
        cv2.polylines(blank,[np.array(cpts)],False,mark_color,thickness=5)
        h,w = blank.shape
        if (len(np.nonzero(blank[:,:int(w/2)])[0] > len(np.nonzero(blank[:,int(w/2):])[0]))): #is left lane
            lane_masks[0] = blank
        else: #is right lane
            lane_masks[1] = blank

    display_img(curves_img, cmap="", convertToRGB=True)

    #save the image
    cv2.imwrite("predicted_lanes/img" + str(img_number).zfill(4) + "_lanes.jpg", curves_img)

    ### Feed Predictions into Future Steps ###

    #slide over 1 image in image list
    list_of_RGB_Images.pop(0)
    list_of_RGB_Images.append(img)

    #get masks for your predictions. Label according to what predominate mask was in last frame
    h, w,depth = img.shape

    #left
    last_img_mask_w = np.copy(colorMask[0][-int(grayImg.shape[0] * grayImg.shape[1] * crop_pct):])
    last_img_mask_w = last_img_mask_w.reshape(int(grayImg.shape[0]*crop_pct),grayImg.shape[1])
    last_img_mask_w[:,int(w/2):] = 0

    last_img_mask_y = np.copy(colorMask[1][-int(grayImg.shape[0] * grayImg.shape[1] * crop_pct):])
    last_img_mask_y = last_img_mask_y.reshape(int(grayImg.shape[0] * crop_pct), grayImg.shape[1])
    last_img_mask_y[:,int(w/2):] = 0

    left_label = "w" if len(np.nonzero(last_img_mask_w)[0]) > len(np.nonzero(last_img_mask_y)[0]) else "y"

    #right
    last_img_mask_w = np.copy(colorMask[0][-int(grayImg.shape[0] * grayImg.shape[1] * crop_pct):])
    last_img_mask_w = last_img_mask_w.reshape(int(grayImg.shape[0] * crop_pct), grayImg.shape[1])
    last_img_mask_w[:, :int(w/2)] = 0

    last_img_mask_y = np.copy(colorMask[1][-int(grayImg.shape[0] * grayImg.shape[1] * crop_pct):])
    last_img_mask_y = last_img_mask_y.reshape(int(grayImg.shape[0] * crop_pct), grayImg.shape[1])
    last_img_mask_y[:, :int(w/2)] = 0

    right_label = "y" if len(np.nonzero(last_img_mask_w)[0]) < len(np.nonzero(last_img_mask_y)[0]) else "w"

    if left_label == "w" and right_label == "y":
        mask_w = lane_masks[0]
        mask_y = lane_masks[1]
    elif left_label == "y" and right_label == "w":
        mask_w = lane_masks[1]
        mask_y = lane_masks[0]
    elif left_label == "y" and right_label == "y":
        mask_y = cv2.bitwise_and(lane_masks[0],lane_masks[1])
        mask_w = np.zeros_like(lane_masks[0])
    else:
        mask_w = cv2.bitwise_and(lane_masks[0],lane_masks[1])
        mask_y = np.zeros_like(lane_masks[0])

    display_img(mask_w)
    display_img(mask_y)

    #add a depth to each mask because createTrainingData expects it
    mask_w = np.repeat(mask_w[:, :, np.newaxis], 3, axis=2)
    mask_y = np.repeat(mask_y[:, :, np.newaxis], 3, axis=2)

    #pop off first image mask and add this new image mask
    imageMasks_w.pop(0)
    imageMasks_w.append(mask_w)

    imageMasks_y.pop(0)
    imageMasks_y.append(mask_y)

    X, y_w, y_y = createTrainingData(list_of_RGB_Images, imageMasks_w, imageMasks_y)
