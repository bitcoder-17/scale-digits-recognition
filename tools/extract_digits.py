import cv2
from pathlib import Path
import numpy as np
from argparse import ArgumentParser


def binarize_threshold(image: np.ndarray):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, (0, 0, 0), (180, 255, 200))
    return binary


def improve_mask(mask: np.ndarray):
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    for box in bboxes:
        if box[2] > 0.5 * mask.shape[1]:
            x, y, w, h = box
            mask[y: y+h, x: x+w] = 0

    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    bboxes = [cv2.boundingRect(contour) for contour in contours]
    bboxes = sorted(bboxes, key=lambda box: box[0])  # sorted by x
    for i in range(len(bboxes) - 1):
        left_box_x = bboxes[i][0] + bboxes[i][2]    # leftbox.x + leftbox.w
        left_box_y = bboxes[i][1] + bboxes[i][3]
        for next_box in bboxes[i+1: i+6]:
            right_box_x = next_box[0]                # rightbox.x
            right_box_y = next_box[1] + next_box[3]
            if abs(left_box_x - right_box_x) < 0.02 * mask.shape[1]:
                mask[:, left_box_x:right_box_x + 1] = 255
                if abs(left_box_y - right_box_y) < 0.02 * mask.shape[0]:
                    mask[left_box_y:right_box_y, left_box_x:right_box_x + 1] = 255

    return mask


def character_segmentation(image: np.ndarray):
    margin_row = int(image.shape[0] * 0.05)
    margin_col = int(image.shape[1] * 0.075)

    image = image[margin_row:image.shape[0] - margin_row, margin_col:image.shape[1] - margin_col]
    binary = binarize_threshold(image)
    binary = improve_mask(binary)

    count_col = np.count_nonzero(binary, axis=0)
    count_col = count_col / binary.shape[0]
    count_row = np.count_nonzero(binary, axis=1)
    count_row = count_row / binary.shape[1]

    mask_col = np.zeros_like(binary)
    mask_col[:, count_col > 0.03] = 255.

    mask_color = np.zeros_like(image)
    mask_color[mask_col != 0] = (0, 0, 255)

    # alpha = 0.3
    # blend = cv2.addWeighted(image, alpha, mask_color, 1 - alpha, 1)
    # cv2.imshow('Segmentation', blend)

    contours = cv2.findContours(mask_col, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    chars = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rect = (x + margin_col, y + margin_row, w, h)
        chars.append(rect)
    return chars


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--ext', type=str, default='png')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    image_paths = sorted(list(input_dir.glob(f'*.{args.ext}')))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        char_bboxes = character_segmentation(image)
        char_bboxes = sorted(char_bboxes, key=lambda box: box[0], reverse=True)
        for i, (x, y, w, h) in enumerate(char_bboxes):
            crop = image[y:y+h, x:x+w]
            output_path = output_dir.joinpath(f'{image_path.stem}_{i}.jpg')
            cv2.imwrite(str(output_path), crop)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)

        if args.visualize:
            cv2.imshow('Digits', image)
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                break

    cv2.destroyAllWindows()
    print('Done')
