from descriptors.circularity import circularity
import cv2
import numpy as np
from warnings import warn

from descriptors.utils import moments as m


def compactness(image, method='perimeters_ratio', approx_contour=True):
    if method == 'perimeters_ratio':
        return circularity(image, method='Cst', approx_contour=approx_contour)
    elif method == 'p2a':
        _, contours, _ = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            warn("More than one blob?")
        cnt = contours[0]

        if approx_contour:
            cnt = cv2.approxPolyDP(cnt, epsilon=1, closed=True)
        perimeter = cv2.arcLength(cnt, closed=True)
        area = cv2.contourArea(cnt)

        return CalculateCompactness.perimeter_to_area(perimeter=perimeter, area=area)
    elif method == 'shape_to_min_circle_at_shape':
        _, contours, _ = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            warn("More than one blob?")
        cnt = contours[0]

        if approx_contour:
            cnt = cv2.approxPolyDP(cnt, epsilon=1, closed=True)
        shape_area = cv2.contourArea(cnt)

        (_, _), min_enclosing_radius = cv2.minEnclosingCircle(cnt)
        min_enclosing_circle_area = np.pi * (min_enclosing_radius ** 2)

        return CalculateCompactness.shape_to_min_circle_at_shape(shape_area=shape_area, minimal_circle_area=min_enclosing_circle_area)
    elif method == 'perimeter_circle_to_perimeter_shape_fields_equal':
        shape_area = m.m00(image)
        _, contours, _ = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1:
            warn("More than one blob?")
        cnt = contours[0]

        if approx_contour:
            cnt = cv2.approxPolyDP(cnt, epsilon=1, closed=True)
        shape_perimeter = cv2.arcLength(cnt, closed=True)
        return CalculateCompactness.perimeter_circle_to_perimeter_shape_fields_equal(shape_area=shape_area, shape_perimeter=shape_perimeter)


class CalculateCompactness(object):
    """
        Methods to calculate compactness from given parameters
        Based on paper https://www.researchgate.net/profile/Ernesto_Bribiesca/publication/228948093_State_of_the_art_of_compactness_and_circularity_measures/links/0deec5339984009465000000/State-of-the-art-of-compactness-and-circularity-measures.pdf
    """

    @staticmethod
    def perimeter_to_area(perimeter, area):
        return (perimeter ** 2) / area

    @staticmethod
    def isoperimetric_ratio(perimeter, area):
        return (4 * np.pi * area) / (perimeter ** 2)

    @staticmethod
    def shape_to_min_circle_at_shape(shape_area, minimal_circle_area):
        return shape_area / minimal_circle_area

    @staticmethod
    def perimeter_circle_to_perimeter_shape_fields_equal(shape_area, shape_perimeter):
        circle_r = np.sqrt(shape_area / np.pi)
        circle_perimeter = 2 * np.pi * circle_r
        return circle_perimeter / shape_perimeter
