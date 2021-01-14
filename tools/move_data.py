from pathlib import Path
from argparse import ArgumentParser
import shutil


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('--ext', type=str, default='png')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    image_files = input_dir.glob(f'*.{args.ext}')
    for image_file in image_files:
        for dir_path in input_dir.iterdir():
            files = dir_path.glob(f'*.{args.ext}')
            if image_file.name in [class_file.name for class_file in files]:
                shutil.move(str(image_file), str(dir_path.joinpath(image_file.name)))
                break
