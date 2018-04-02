import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import sys
from config import *

def drawPixel(x, y, pixelData, width, height):
    if x < height and y < width:
        pixelData[x,y] += 1
    return pixelData


def paint_pixels(x, y, image):
    image[x,y,1] = 255
    image[x,y,0] = 0
    image[x,y,2] = 0
    return image

def draw_circle(x0, y0, radius, image):
    ''' Draw a circle using the provided radius and circle center information '''
    x = radius;
    y = 0;
    decisionOver2 = 1 - x;   # Decision criterion divided by 2 evaluated at x=r, y=0

    while x >= y:
        image = paint_pixels(x + x0, y + y0, image)
        image = paint_pixels(y + x0, x + y0, image)
        image = paint_pixels(-x + x0, y + y0, image)
        image = paint_pixels(-y + x0, x + y0, image)
        image = paint_pixels(-x + x0, -y + y0, image)
        image = paint_pixels(-y + x0, -x + y0, image)
        image = paint_pixels(x + x0, -y + y0, image)
        image = paint_pixels(y + x0, -x + y0, image)
        y+=1
        if decisionOver2 <= 0:
            decisionOver2 += 2 * y + 1; # Change in decision criterion for y -> y+1
        else:
            x-=1
            decisionOver2 += 2 * (y - x) + 1; # Change for y -> y+1, x -> x-1
    return image

def accumulator_data(x0, y0, radius, pixelData, width, height):
    ''' This is the implementation of Midpoint Circle Algorithm. Refer the report for more details. '''
    x = radius;
    y = 0;
    decisionOver2 = 1 - x;   # Decision criterion divided by 2 evaluated at x=r, y=0

    while x >= y:
        pixelData = drawPixel(x + x0, y + y0, pixelData, width, height)
        pixelData = drawPixel(y + x0, x + y0, pixelData, width, height)
        pixelData = drawPixel(-x + x0, y + y0, pixelData, width, height)
        pixelData = drawPixel(-y + x0, x + y0, pixelData, width, height)
        pixelData = drawPixel(-x + x0, -y + y0, pixelData, width, height)
        pixelData = drawPixel(-y + x0, -x + y0, pixelData, width, height)
        pixelData = drawPixel(x + x0, -y + y0, pixelData, width, height)
        pixelData = drawPixel(y + x0, -x + y0, pixelData, width, height)
        y+=1
        if decisionOver2 <= 0:
            decisionOver2 += 2 * y + 1; # Change in decision criterion for y -> y+1
        else:
            x-=1
            decisionOver2 += 2 * (y - x) + 1; # Change for y -> y+1, x -> x-1
    return pixelData

def get_edge_locations(edged_image):
    edges = np.where(edged_image==255)
    return edges

def get_max_possible_radius(edges, max_detection_radius):
    ''' Optimization technique to reduce the different radii to scan while generating accumulator array ''' 
    xmin = min(edges[0])
    xmax = max(edges[0])

    ymin = min(edges[1])
    ymax = max(edges[1])

    return min(min(xmax-xmin, ymax-ymin)/2,max_detection_radius)

def construct_accumulator_array(edges, width, height, max_radius):
    ''' Initialize an accumulator array and generate the accumulator array by using the detected edges '''
    acc_array = np.zeros(((height,width,max_radius)))
    for radius in range(5, max_radius):
        for i in xrange(0,len(edges[0])):
            x=edges[0][i]
            y=edges[1][i]
            acc_array[:,:,radius] = accumulator_data(x,y,radius, acc_array[:,:,radius], width, height)
    return acc_array

def threshold_accumulator_plot_circles(output, acc_array, max_radius, width, height, intensity_threshold, accumulator_kernel):
    ''' Detecting the circle centers and radius from the accumulator array. Also handles overlaying the 
        detected circles on the original image. '''
    i, j = 0, 0
    filter_patch = accumulator_kernel
    center_loc_filter = np.ones((filter_patch,filter_patch,max_radius))
    while(i<height-filter_patch):
        while(j<width-filter_patch):
            center_loc_filter=acc_array[i:i+filter_patch,j:j+filter_patch,:] * center_loc_filter
            max_pt = np.where(center_loc_filter==center_loc_filter.max())
            x0 = max_pt[0]       
            y0 = max_pt[1]
            radius = max_pt[2]
            y0 += j
            x0 += i
            if(center_loc_filter.max()>intensity_threshold):
                try:
                    if len(x0) > 1 and len(y0) > 1 and len(radius) > 1:
                        x0 = x0[0]
                        y0 = y0[0]
                        radius = radius[0]
                    output = draw_circle(x0, y0, radius, output)
                except Exception as e:
                    print e

            j=j+filter_patch
            center_loc_filter[:,:,:]=1
        j=0
        i=i+filter_patch
    return output

def main(filename, intensity_threshold, accumulator_kernel, gaussian_kernel, max_detection_radius, output_file):
    ''' Main function knitting all sub tasks of the hough circle algorithm'''
    original_image = cv2.imread(filename,1)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    output = original_image.copy()

    #Gaussian Blurring of Gray Image
    blur_image = cv2.GaussianBlur(gray_image,(gaussian_kernel,gaussian_kernel),0)
    #Using OpenCV Canny Edge detector to detect edges
    edged_image = cv2.Canny(blur_image,75,150)
    height,width = edged_image.shape

    edges = get_edge_locations(edged_image)
    max_radius = get_max_possible_radius(edges, max_detection_radius)
    acc_array = construct_accumulator_array(edges, width, height, max_radius)
    output = threshold_accumulator_plot_circles(output, acc_array, max_radius, width, height, intensity_threshold, accumulator_kernel)  
    cv2.imwrite(output_file,output)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Please provide appropriate parameters as follows: python hough_transform.py input_file_name center_threshold"
        sys.exit()

    input_file_name = str(sys.argv[1])
    intensity_threshold = float(sys.argv[2])
    main(input_file_name, intensity_threshold, accumulator_kernel, gaussian_kernel, max_detection_radius, output_file)