import base64
import copy
import json
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image


if __name__ == "__main__":
    json_dict_template = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [
            {
                "label": "digits",
                "points": [
                    [
                        614.0672782874618,
                        553.8226299694189
                    ],
                    [
                        727.5229357798165,
                        543.1192660550458
                    ],
                    [
                        729.9694189602446,
                        596.0244648318043
                    ],
                    [
                        616.8195718654434,
                        609.480122324159
                    ]
                ],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
        ],
        "imagePath": "",
        "imageData": None,
        "imageHeight": 1080,
        "imageWidth": 1440
    }

    parser = ArgumentParser()
    parser.add_argument('frame_dir', type=str, help='Directory where the frames are located in')
    parser.add_argument('--ext', default='png', help='Image extension')
    args = parser.parse_args()

    frames = Path(args.frame_dir).glob(f'*.{args.ext}')
    for frame_path in frames:
        json_dict = copy.deepcopy(json_dict_template)
        json_dict['imagePath'] = frame_path.name
        encoded_image = base64.b64encode(open(frame_path, "rb").read())
        json_dict['imageWidth'], json_dict['imageHeight'] = Image.open(frame_path).size
        json.dump(json_dict, open(frame_path.with_suffix('.json'), 'wt', encoding='utf8'), ensure_ascii=False, indent=4)
