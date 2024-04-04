#!/usr/bin/python

import sys
import cv2
import numpy as np
import math
import struct
from datetime import datetime
import glob

WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
SIZE = (80, 80)

ALL_TEMPLATES = [cv2.imread(template) for template in sorted(glob.glob("templates/*.png"))]
ALL_TEMPLATES = [cv2.resize(template, SIZE) for template in ALL_TEMPLATES]

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, one_c):
    #https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    
    pts = [((float(one_c[0])), float(one_c[1])),
            ((float(one_c[2])), float(one_c[3])),
            ((float(one_c[4])), float(one_c[5])),
            ((float(one_c[6])), float(one_c[7]))]
    
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	    [0, 0],
	    [maxWidth - 1, 0],
	    [maxWidth - 1, maxHeight - 1],
	    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def mean(data):
    return sum(data) / len(data)

def pearson_correlation_coefficient(x, y):
    # Calculate means
    mean_x = mean(x)
    mean_y = mean(y)
    
    # Calculate numerator and denominators
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x)
    denominator_y = sum((yi - mean_y) ** 2 for yi in y)
    
    # Calculate correlation coefficient
    correlation_coefficient = numerator / ((denominator_x * denominator_y) ** 0.5)
    
    return correlation_coefficient

def calculate_accuracy_and_fscore(ground_truth, detected: list):
    tp = fp = tn = fn = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] == 0 and detected[i][0] == 0 or ground_truth[i] == 1 and detected[i][0] == 1:
            tp += 1
        # elif ground_truth[i] == 1 and detected[i][0] == 1:
        #     tn += 1
        elif ground_truth[i] == 0 and detected[i][0] == 1:
            fp += 1
            #cv2.imwrite(f'template_00{i}.png', detected[i][1])
        elif ground_truth[i] == 1 and detected[i][0] == 0:
            fn += 1

    if tp == 0 or (tp + fp) == 0 or (tp + fn) == 0:
        return 0, 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, fscore

def match_template(img, templates: list):
    threshold = 0.06

    for template in templates:
        result = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if min_val < threshold:
            return True, min_loc
        
    return False, None

def load_ground_truth_data() -> list:
    ground_truth_data = []
    for file in sorted(glob.glob("test_images_zao/*.txt")):
        with open(file, 'r') as f:
            lines = f.readlines()
            current_line = [int(line.strip()) for line in lines]
            ground_truth_data.append(current_line)
    return ground_truth_data

def template_matching_detection(test_images, pkm_coordinates, ground_truth_data):
    scores = []
    
    for img_name, ground_truth in zip(test_images, ground_truth_data):
        image = cv2.imread(img_name)
        image_clone = image.copy()
        n_park = 0
        empty_spaces_detected = []
        for coord in pkm_coordinates:
            point = order_points(np.array([[float(coord[0]), float(coord[1])],
                                           [float(coord[2]), float(coord[3])],
                                           [float(coord[4]), float(coord[5])],
                                           [float(coord[6]), float(coord[7])]]))
            center = np.mean(point, axis=0)

            one_place_img = four_point_transform(image, coord)
            one_place_img = cv2.resize(one_place_img, SIZE)

            is_empty_space, max_loc = match_template(one_place_img, ALL_TEMPLATES)
            if is_empty_space:
                empty_spaces_detected.append((0, one_place_img))
                cv2.putText(image_clone, str(n_park + 1), (int(center[0] + 10), int(center[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
                cv2.circle(image_clone, (int(center[0]), int(center[1])), 8, GREEN, -1)
            else:
                empty_spaces_detected.append((1, one_place_img))
                cv2.putText(image_clone, str(n_park + 1), (int(center[0] + 10), int(center[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
                cv2.circle(image_clone, (int(center[0]), int(center[1])), 8, RED, -1)

            n_park += 1
        accuracy, fscore = calculate_accuracy_and_fscore(ground_truth, empty_spaces_detected)
        scores.append((accuracy, fscore))
        #print(f'Accuracy: {accuracy}, F-score: {fscore}')

        # cv2.imshow('Detected Parking Spaces', image_clone)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print(f'Average accuracy: {mean([score[0] for score in scores])}, Average F-score: {mean([score[1] for score in scores])}')

def sobel_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    gradient = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    _, edges = cv2.threshold(gradient, 80, 255, cv2.THRESH_BINARY)
    
    return edges

def canny_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 80, 140)
    return edges

def laplacian_detect(image):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  laplacian = cv2.Laplacian(gray, cv2.CV_64F)
  laplacian_abs = cv2.convertScaleAbs(laplacian)
  _, edges = cv2.threshold(laplacian_abs, 40, 100, cv2.THRESH_BINARY)

  return edges

def edge_detector(test_images, pkm_coordinates, ground_truth_data, method_n):
    scores = []
    
    for img_name, ground_truth in zip(test_images, ground_truth_data):
        image = cv2.imread(img_name)
        image_clone = image.copy()
        
        n_park = 0
        empty_spaces_detected = []
        for coord in pkm_coordinates:
            point = order_points(np.array([[float(coord[0]), float(coord[1])],
                                           [float(coord[2]), float(coord[3])],
                                           [float(coord[4]), float(coord[5])],
                                           [float(coord[6]), float(coord[7])]]))
            center = np.mean(point, axis=0)

            one_place_img = four_point_transform(image, coord)
            one_place_img = cv2.resize(one_place_img, SIZE)

            if method_n == 0:
                edges = sobel_detect(one_place_img)
                threshold = 330
            elif method_n == 1:
                edges = laplacian_detect(one_place_img)
                threshold = 320
            elif method_n == 2:
                edges = canny_detect(one_place_img)
                threshold = 320

            nonzero_count = np.count_nonzero(edges)

            if nonzero_count < threshold:
                empty_spaces_detected.append((0, one_place_img))
                cv2.putText(image_clone, str(n_park + 1), (int(center[0] + 10), int(center[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
                cv2.circle(image_clone, (int(center[0]), int(center[1])), 8, GREEN, -1)
            else:
                empty_spaces_detected.append((1, one_place_img))
                cv2.putText(image_clone, str(n_park + 1), (int(center[0] + 10), int(center[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
                cv2.circle(image_clone, (int(center[0]), int(center[1])), 8, RED, -1)

            n_park += 1
        accuracy, fscore = calculate_accuracy_and_fscore(ground_truth, empty_spaces_detected)
        scores.append((accuracy, fscore))
        # print(f'Accuracy: {accuracy}, F-score: {fscore}')

        # cv2.imshow('Detected Parking Spaces', image_clone)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print(f'Average accuracy: {mean([score[0] for score in scores])}, Average F-score: {mean([score[1] for score in scores])}')

def main(argv):
    test_images = sorted(glob.glob("test_images_zao/*.jpg"))

    ground_truth_data = load_ground_truth_data()

    pkm_coordinates = []
    with open('parking_map_python.txt', 'r') as pkm_file:
        for line in pkm_file.readlines():
            sp_line = line.strip().split(" ")
            pkm_coordinates.append(sp_line)


    print("Template Matching: ")
    template_matching_detection(test_images, pkm_coordinates,ground_truth_data)
    print("Sobel Edge Detection: ")
    edge_detector(test_images, pkm_coordinates, ground_truth_data, 0)
    print("Laplacian Edge Detection: ")
    edge_detector(test_images, pkm_coordinates, ground_truth_data, 1)
    print("Canny Edge Detection: ")
    edge_detector(test_images, pkm_coordinates, ground_truth_data, 2)

if __name__ == "__main__":
    main(sys.argv[1:])


    