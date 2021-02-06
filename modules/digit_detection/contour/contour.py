from typing import List
import numpy as np
import cv2
# from shapely import geometry, ops
import imutils


class ImplContour(object):
    def __init__(self):
        self.min_area = 25

    def run(self, image: np.ndarray, mask: np.ndarray) -> List[str]:
        bboxes, botboxes = self.character_segmentation_contours(mask)
        for x, y, w, h in bboxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        for x, y, w, h in botboxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Digits', image)

    def character_segmentation_contours(self, mask: np.ndarray):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        keep_contour_indexes = [i for i in range(len(contours)) if hierarchy[0][i][-1] == -1]
        contours = [contours[i] for i in keep_contour_indexes]
        hierarchy = np.array([[hierarchy[0][i] for i in keep_contour_indexes]])

        # contours = imutils.grab_contours((contours, hierarchy))
        boxes = []
        for contour in contours:
            box = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area >= self.min_area:
                boxes.append(box)

        topleft_points = [(box[0], box[1]) for box in boxes]
        self.min_y = 10
        self.max_y = 25

        topboxes = [box for i, box in enumerate(boxes) if self.min_y <= topleft_points[i][1] <= self.max_y]
        not_topboxes = [box for box in boxes if box not in topboxes]


        return topboxes, not_topboxes
        chars = []
        return chars