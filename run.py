from modules.digit_recognition.infer import DigitRecognizer
from modules.objects import FrameInfo
# from modules.digit_recognition.recognize import DigitRecognizer
from modules.digit_detection.infer import DigitDetection
from modules.field_extraction.infer import FieldExtractor
from pathlib import Path
from modules.segmentation.infer import Segmentor
from argparse import ArgumentParser
import cv2


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--ext', type=str, default='png')
    args = parser.parse_args()

    field_extractor = FieldExtractor(128, 96)

    feature_weight = Path('./weights/segmentor/feature_encoder_499_1_way_10shot.pkl')
    relation_weight = Path('./weights/segmentor/relation_network_499_1_way_10shot.pkl')
    segmentor = Segmentor(feature_weight=feature_weight, relation_weight=relation_weight)
    digit_detector = DigitDetection('fixed')
    digit_recognizer = DigitRecognizer('cnn')

    for image_path in args.input_dir.glob(f'*.{args.ext}'):
        print(f'Processing {image_path}')
        image = cv2.imread(str(image_path))

        info = FrameInfo(str(image_path), image.copy())

        field_info = field_extractor.run(image)
        if field_info is None:
            print('---> Could not extract field')
        else:
            field_info = segmentor.run(field_info)
            field_info = digit_detector.run(field_info)
            field_info = digit_recognizer.run(field_info)
            # print('Value:', field_info.value)
            cv2.imshow('Image', field_info.visualize())
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27:
                break

    cv2.destroyAllWindows()
