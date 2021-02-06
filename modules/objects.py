from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import cv2


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def from_dict(d: Dict) -> 'Point':
        return Point(d['x'], d['y'])

    @staticmethod
    def from_list(l: Union[Tuple[int, int]]) -> 'Point':
        return Point(l[0], l[1])


class Box(object):
    def __init__(self, points: List[Point]):
        assert len(points) == 2
        self.points = points
        self.text = None

    @property
    def text(self):
        return self.__text

    @text.setter
    def text(self, value):
        self.__text = str(value)

    def sort_points(self) -> 'Box':
        min_x = min(self.points[0].x, self.points[1].x)
        min_y = min(self.points[0].y, self.points[1].y)
        max_x = max(self.points[0].x, self.points[1].x)
        max_y = max(self.points[0].y, self.points[1].y)
        return Box([Point(min_x, min_y), Point(max_x, max_y)])


class FieldInfo(object):
    def __init__(self, image: np.ndarray, mask_image: Optional[np.ndarray] = None, boxes: List[Box] = [], mask_feature: Optional[torch.Tensor] = None):
        self.image = image
        self.mask_image = mask_image
        self.mask_feature = mask_feature
        self.boxes = boxes

    @property
    def value(self):
        # for box in self.boxes:
        #     print(box.text)
        values = ''.join([box.text for box in self.boxes if box.text is not None])
        return values

    def visualize(self):
        draw_image = self.image.copy()
        for box in self.boxes:
            cv2.rectangle(draw_image, (box.points[0].x, box.points[0].y), (box.points[1].x, box.points[1].y), (0, 0, 255), 1)
            cv2.putText(draw_image, box.text, (box.points[0].x, box.points[1].y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return draw_image


class FrameInfo(object):
    def __init__(self,
                 frame_name: str,
                 frame_image: np.ndarray,
                #  roi_points: List[Point],
                 field_info: Optional[FieldInfo] = None):
        self.frame_name = frame_name
        self.frame_image = frame_image
        # self.roi_points = roi_points
        self.field_info = field_info

    @property
    def value(self):
        if self.field_info is None:
            return ''
        return self.field_info.value
