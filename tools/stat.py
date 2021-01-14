from pathlib import Path
import pandas as pd
from argparse import ArgumentParser
from PIL import Image


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--ext', type=str, default='png')
    args = parser.parse_args()

    image_paths = {}
    for image_path in Path(args.input_dir).glob(f'**/*.{args.ext}'):
        class_ = image_path.parent.stem
        image_paths[class_] = image_paths.get(class_, [])
        image_paths[class_].append(image_path)

    stat = {}
    for class_, paths in image_paths.items():
        sum_h, sum_w = 0, 0
        for path in paths:
            image = Image.open(path)
            w, h = image.size
            sum_h += h
            sum_w += w
        mean_w = sum_w / len(paths)
        mean_h = sum_h / len(paths)
        stat[class_] = {
            'mean_w': mean_w,
            'mean_h': mean_h,
            'w/h': mean_w / mean_h,
            'num_samples': len(paths),
        }

    size_df = pd.DataFrame(data=stat).transpose()
    if args.output_file is not None:
        size_df.to_csv(args.output_file)
    else:
        print(size_df)