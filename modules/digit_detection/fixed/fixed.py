from modules.objects import Box, FieldInfo, Point
from typing import List
import numpy as np
import cv2


class ImplFixed(object):
    def __init__(self):
        self.right_offset = 6
        self.top_offset = 10
        self.width = 20
        self.height = 70
        self.overlap = 3
        self.ratio = 0.03

    def run(self, field_info: FieldInfo) -> FieldInfo:
        bboxes = self.character_segmentation(field_info.mask_image)

        # TODO: test only
        # image = field_info.image.copy()
        boxes = []
        for x, y, w, h in bboxes:
            box = Box([Point(x, y), Point(x + w, y + h)])
            box = box.sort_points()
            boxes.append(box)
            # cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # cv2.imshow('Digits', image)

        field_info.boxes = boxes
        return field_info

    def character_segmentation(self, mask: np.ndarray):
        WIDTH = mask.shape[1]
        chars = []
        for i in range(1, 8):
            x = WIDTH - self.right_offset - self.width * i + self.overlap * (i-1)
            y = self.top_offset
            w = self.width
            h = self.height

            non_zero = cv2.countNonZero(mask[y:y+h, x:x+w])
            ratio = non_zero / (w * h)
            print(i, ratio)
            if ratio >= self.ratio:
                chars.append((x, y, w, h))
            else:
                break
        chars = sorted(chars, key=lambda box: box[0])
        return chars
