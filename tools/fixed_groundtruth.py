import copy
import json
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('ref_json', type=str, help='A reference json file to duplicate for other images')
    parser.add_argument('frame_dir', type=str, help='Directory where the frames are located in')
    parser.add_argument('--ext', default='png', help='Image extension')
    args = parser.parse_args()

    json_template = json.load(open(args.ref_json, 'rt', encoding='utf-8'))

    frames = Path(args.frame_dir).glob(f'*.{args.ext}')
    for frame_path in frames:
        print(f'Processing {frame_path}')
        json_dict = copy.deepcopy(json_template)
        json_dict['imagePath'] = frame_path.name
        json_dict['imageWidth'], json_dict['imageHeight'] = Image.open(frame_path).size
        json_dict['imageData'] = None
        json.dump(json_dict, open(frame_path.with_suffix('.json'), 'wt', encoding='utf8'), ensure_ascii=False, indent=4)
