from pathlib import Path
from argparse import ArgumentParser
import cv2


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('input_ext', type=str, help='Image extension to map from, e.g. jpg')
    parser.add_argument('output_ext', type=str, help='Image extension to map to, e.g. png')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    file_iter = input_dir.glob(f'**/*.{args.input_ext}')
    for file_path in file_iter:
        image = cv2.imread(str(file_path))
        output_path = Path(str(file_path).replace(str(input_dir), str(output_dir))).with_suffix(f'.{args.output_ext}')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_ok = cv2.imwrite(str(output_path), image)
        if is_ok:
            print('Converted from {} to {}'.format(str(file_path), str(output_path)))
