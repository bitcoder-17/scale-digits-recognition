import cv2
from pathlib import Path
from argparse import ArgumentParser
import json


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--ext', type=str, default='png')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    json_paths = sorted(list(input_dir.glob(f'*.json')))

    for json_path in json_paths:
        print(f'Extracting from {json_path}')
        json_dict = json.load(open(json_path, 'rt'))
        image_path = json_path.with_suffix(f'.{args.ext}')
        image = cv2.imread(str(image_path))

        json_dict['shapes'] = sorted(json_dict['shapes'], key=lambda shape: min(shape['points'][0][0], shape['points'][1][0]))
        for i, shape in enumerate(json_dict['shapes']):
            p1, p2 = shape['points']
            x1, x2 = int(min(p1[0], p2[0])), int(max(p1[0], p2[0]))
            y1, y2 = int(min(p1[1], p2[1])), int(max(p1[1], p2[1]))
            crop = image[y1:y2, x1:x2]
            output_path = output_dir.joinpath(f'{json_path.stem}_{i}.{args.ext}')
            cv2.imwrite(str(output_path), crop)

    cv2.destroyAllWindows()
    print('Done')
