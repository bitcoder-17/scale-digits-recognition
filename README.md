# Scaler Digits Recognition

# Run
1. To extract from frame 1000th to 2000th and keep one frame per 10 frames from a video
```
python tools/extract_frames.py input_video output_dir --start 1000 --end 2000 --skip 10
```

2. To extract digits from `labelme`:
```
python tools/extract_digits_label.py input_dir output_dir --ext png
```
where `input_dir` is the directory containing the `JSON` files and `output_dir` is the directory where the digits are extracted to.

3. To extract the content within the polygon box after the 'labelme' step:
```
python extract_textline.py input_dir output_dir
```
where `input_dir` is the directory containing images and json files(after the 'labelme' boxing step) and `output_dir` is the directory where the textlines are extracted to.

4. Fixed Groundtruth:
```
python tools/fixed_groundtruth.py ref_json frame_dir
```
where `ref_json` is the reference `JSON` files and `frame_dir` is the directory which contains all the images and json files.

