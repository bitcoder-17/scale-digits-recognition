import cv2
from pathlib import Path
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_video', type=str, help='Input video path')
    parser.add_argument('output_dir', type=str, help='Output directory where the frames are extracted to')
    parser.add_argument('--skip', type=int, default=10, help='The number of frames to skip, e.g. 10')
    parser.add_argument('--start', '-s', type=int, default=0, help='The start frame index')
    parser.add_argument('--end', '-e', type=int, default=None, help='The end frame index')
    parser.add_argument('--ext', type=str, default='png', help='Image extension')

    args = parser.parse_args()

    video_path = Path(args.input_video)
    output_dir = Path(args.output_dir).joinpath(video_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    skip = args.skip
    ext = args.ext
    begin_idx = args.start
    end_idx = args.end

    video = cv2.VideoCapture(str(video_path))

    i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if begin_idx <= i:  # exclude the last index since we start from zero
            if i >= end_idx:
                break
            if i % skip == 0:
                output_path = output_dir.joinpath(f'{video_path.stem}_{i}.{ext}')
                cv2.imwrite(str(output_path), frame)
                print(f'Wrote to {output_path}')
        i += 1

    video.release()
    print('Done')
