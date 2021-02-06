from modules.objects import FieldInfo
from typing import List
import numpy as np
import cv2

from .contour.contour import ImplContour
from .fixed.fixed import ImplFixed


class DigitDetection(object):
    def __init__(self, mode: str):
        if mode == 'fixed':
            self.processor = ImplFixed()
        elif mode == 'contour':
            self.processor = ImplContour()
        else:
            raise ValueError(f'Unknow mode = {mode}')

    def run(self, field_info: FieldInfo) -> FieldInfo:
        return self.processor.run(field_info)
