# Scaler Digits Recognition

# Run
1. To extract from frame 1000th to 2000th and keep one frame per 10 frames from a video
```
python tools/extract_frames.py input_video output_dir --start 1000 --end 2000 --skip 10
```

1. To extract digits from `labelme`:
```
python tools/extract_digits_label.py input_dir output_dir --ext png
```
where `input_dir` is the directory containing the `JSON` files and `output_dir` is the directory where the digits are extracted to.