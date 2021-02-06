from typing import Optional
from modules.objects import FieldInfo
import numpy as np
import math
import cv2


class FieldExtractor(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def run(self, image: np.ndarray) -> Optional[FieldInfo]:
        image = self._crop_roi_by_corners(image)
        if image is not None:
            image = cv2.resize(image, dsize=(self.width, self.height))
            info = FieldInfo(image)
            return info
        return None

    def _heuristic_position_corner_filter(self, corners):
        corner_filtered = []
        for x, y in corners:
            if 15 < x < 20 or 80 < x < 110:
                if 15 < y < 25 or 65 < y < 80:
                    corner_filtered.append((int(x), int(y)))
        return corner_filtered

    def _crop_roi_by_corners(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        corners = cv2.cornerHarris(gray, 10, 3, 0.24)
        corners = cv2.dilate(corners, None)
        ret, corners = cv2.threshold(corners, 0.01*corners.max(), 255, 0)
        corners = np.uint8(corners)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
        corners = self._heuristic_position_corner_filter(corners)
        if len(corners) != 4:
            return None

        ret, corners = order_points(corners)
        if not ret:
            return None

        tl, tr, br, bl = corners
        width = int(np.round(max([distance(tl, tr), distance(bl, br)])))
        height = int(np.round(max([distance(tl, bl), distance(tr, br)])))

        dst = np.array([[0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), dst)
        image = cv2.warpPerspective(image, M, (width, height))
        return image


def order_points(points):
    pts = {}
    for x1, y1 in points:
        count_x_larger = 0
        count_x_smaller = 0
        count_y_larger = 0
        count_y_smaller = 0
        for x2, y2 in points:
            if x1 > x2:
                count_x_larger += 1
            elif x1 < x2:
                count_x_smaller += 1
            if y1 > y2:
                count_y_larger += 1
            elif y1 < y2:
                count_y_smaller += 1
        p = (x1, y1)
        if count_x_larger >= 2 and count_y_larger >= 2:
            pts['br'] = p
        elif count_x_smaller >= 2 and count_y_larger >= 2:
            pts['bl'] = p
        elif count_y_smaller >= 2 and count_x_smaller >= 2:
            pts['tl'] = p
        else:
            pts['tr'] = p

    if len(pts.keys()) == 4:
        return True, [pts['tl'], pts['tr'], pts['br'], pts['bl']]

    return False, None


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
