from pathlib import Path
import cv2
import json
import math
import numpy as np
from argparse import ArgumentParser


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


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

    return [pts['tl'], pts['tr'], pts['br'], pts['bl']]


def get_padding_box(points, x_factor, y_factor):
    tl, tr, br, bl = points
    width = int(np.round(max([distance(tl, tr), distance(bl, br)])))
    height = int(np.round(max([distance(tl, bl), distance(tr, br)])))
    padding_x = x_factor * width
    padding_y = y_factor * height
    points2 = [
        [tl[0] - padding_x, tl[1] - padding_y],
        [tr[0] + padding_x, tr[1] - padding_y],
        [br[0] + padding_x, br[1] + padding_y],
        [bl[0] - padding_x, bl[1] + padding_y],
    ]
    return points2


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str,
                        help='Directory where the frame image and the json label be')
    parser.add_argument('output_dir', type=str,
                        help='Directory where the textline would be extracted to')
    parser.add_argument('--ext', type=str, default='png')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsons = list(input_dir.glob('*.json'))
    json_path: Path
    for json_path in jsons:
        label_dict = json.load(open(json_path, 'rt'))
        if len(label_dict['shapes']) == 0:
            continue

        frame = cv2.imread(str(json_path.with_suffix(f'.{args.ext}')))

        for i, shape in enumerate(label_dict['shapes']):
            points = order_points(shape['points'])
            tl, tr, br, bl = points
            width = int(np.round(max([distance(tl, tr), distance(bl, br)])))
            height = int(np.round(max([distance(tl, bl), distance(tr, br)])))

            dst = np.array([[0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1],
                            [0, height - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(np.array(points, dtype=np.float32), dst)
            warp = cv2.warpPerspective(frame, M, (width, height))

            output_path = output_dir.joinpath(json_path.stem + f'.{args.ext}')
            cv2.imwrite(str(output_path), warp)
