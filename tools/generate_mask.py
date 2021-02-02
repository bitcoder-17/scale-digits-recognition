from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont, ImageQt
from pathlib import Path
from argparse import ArgumentParser
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIntValidator, QKeySequence, QPixmap
import numpy as np
import cv2
import json
import math
from PyQt5.QtWidgets import QAction, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QApplication, QSizePolicy, QVBoxLayout, QWidget


def generate(font_path: Path, font_size: int, text: str,
             bg_color: Tuple[int, int, int], fg_color: Tuple[int, int, int],
             image_size: Optional[Tuple[int, int]] = None,
             pad: Tuple[int, int, int, int] = (0, 0, 0, 0)):
    '''
    Parameters:
    -----------
    - `padding`: (l, t, r, b) 
    '''
    font = ImageFont.truetype(str(font_path), font_size)
    w, h = image_size or font.getsize(text)
    white = Image.new('RGB', (w + pad[0] + pad[2], h + pad[1] + pad[3]), bg_color)
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(white)
    draw.text((pad[0], pad[1]), text, fill=fg_color, font=font, anchor='mr')
    return white


def heuristic_position_corner_filter(corners):
    corner_filtered = []
    for x, y in corners:
        if 15 < x < 20 or 80 < x < 110:
            if 15 < y < 25 or 65 < y < 80:
                corner_filtered.append((int(x), int(y)))
    return corner_filtered


def order_points(points):
    pts = {}
    for x1, y1 in points:
        count_x_larger = 0
        count_x_smaller = 0
        count_y_larger = 0
        count_y_smaller = 0
        for x2, y2 in points:
            if x1 > x2:
                count_x_larger += 1
            elif x1 < x2:
                count_x_smaller += 1
            if y1 > y2:
                count_y_larger += 1
            elif y1 < y2:
                count_y_smaller += 1
        p = (x1, y1)
        if count_x_larger >= 2 and count_y_larger >= 2:
            pts['br'] = p
        elif count_x_smaller >= 2 and count_y_larger >= 2:
            pts['bl'] = p
        elif count_y_smaller >= 2 and count_x_smaller >= 2:
            pts['tl'] = p
        else:
            pts['tr'] = p

    if len(pts.keys()) == 4:
        return True, [pts['tl'], pts['tr'], pts['br'], pts['bl']]

    return False, None


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def crop_roi_by_corners(pilImage):
    image = np.array(pilImage)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    corners = cv2.cornerHarris(gray, 10, 3, 0.24)
    corners = cv2.dilate(corners, None)
    ret, corners = cv2.threshold(corners, 0.01*corners.max(), 255, 0)
    corners = np.uint8(corners)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
    corners = heuristic_position_corner_filter(corners)
    if len(corners) != 4:
        return False, None

    ret, corners = order_points(corners)
    if not ret:
        return False, None
    # for x, y in corners:
    #     x, y = int(x), int(y)
    #     cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
    # cv2.imshow('Corner', image)

    tl, tr, br, bl = corners
    width = int(np.round(max([distance(tl, tr), distance(bl, br)])))
    height = int(np.round(max([distance(tl, bl), distance(tr, br)])))

    dst = np.array([[0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), dst)
    image = cv2.warpPerspective(image, M, (width, height))
    image = Image.fromarray(image)
    return True, image


def generate_mask(pilImage, number_text: str):
    num_digits = 7
    if number_text.isnumeric():
        number_text = f'{int(number_text):{num_digits}d}'
    image = generate(args.font_path, args.font_size, number_text, (0, 0, 0),
                     (255, 255, 255), args.size, pad=(45, 20, 30, 20))
    # resized_input = input_image.resize(image.size)
    resized_image = image.resize(pilImage.size)
    return resized_image

    # # input_image.show()
    # # resized_image.show()
    # # image.show()
    # resized_input.show()
    # Image.blend(image, resized_input, 0.5).show()


class Config(QWidget):
    leftPad = pyqtSignal(str)
    rightPad = pyqtSignal(str)
    topPad = pyqtSignal(str)
    botPad = pyqtSignal(str)
    number = pyqtSignal(str)

    def __init__(self, parent=None):
        super(Config, self).__init__(parent)
        layout = QVBoxLayout(self)

        numberLabel = QLabel('Number')
        self.numberLineEdit = QLineEdit()
        self.numberLineEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.numberLineEdit.textChanged.connect(lambda text: self.number.emit(text))
        layout.addWidget(numberLabel)
        layout.addWidget(self.numberLineEdit)

        leftPadLabel = QLabel('Left')
        self.leftPadLineEdit = QLineEdit()
        self.leftPadLineEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.leftPadLineEdit.textChanged.connect(lambda text: self.leftPad.emit(text))
        self.leftPadLineEdit.setValidator(QIntValidator(0, 999, self.leftPadLineEdit))
        layout.addWidget(leftPadLabel)
        layout.addWidget(self.leftPadLineEdit)

        rightPadLabel = QLabel('Right')
        self.rightPadLineEdit = QLineEdit()
        self.rightPadLineEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.rightPadLineEdit.setValidator(QIntValidator(0, 999, self.rightPadLineEdit))
        self.rightPadLineEdit.textChanged.connect(lambda text: self.rightPad.emit(text))
        layout.addWidget(rightPadLabel)
        layout.addWidget(self.rightPadLineEdit)

        topPadLabel = QLabel('Top')
        self.topPadLineEdit = QLineEdit()
        self.topPadLineEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.topPadLineEdit.setValidator(QIntValidator(0, 999, self.topPadLineEdit))
        self.topPadLineEdit.textChanged.connect(lambda text: self.topPad.emit(text))
        layout.addWidget(topPadLabel)
        layout.addWidget(self.topPadLineEdit)

        botPadLabel = QLabel('Bottom')
        self.botPadLineEdit = QLineEdit()
        self.botPadLineEdit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.botPadLineEdit.setValidator(QIntValidator(0, 999, self.botPadLineEdit))
        self.botPadLineEdit.textChanged.connect(lambda text: self.botPad.emit(text))
        layout.addWidget(botPadLabel)
        layout.addWidget(self.botPadLineEdit)

    def setPad(self, pad: Tuple[int, int, int, int]):
        l, t, r, b = list(map(str, pad))
        self.leftPadLineEdit.setText(l)
        self.rightPadLineEdit.setText(r)
        self.topPadLineEdit.setText(t)
        self.botPadLineEdit.setText(b)

    def setNumber(self, number):
        self.numberLineEdit.setText(number)

    @pyqtSlot()
    def softClear(self):
        self.numberLineEdit.clear()

class ImageView(QWidget):
    def __init__(self, parent=None):
        super(ImageView, self).__init__(parent)
        layout = QVBoxLayout(self)
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.label)

    @pyqtSlot(Image.Image)
    def setImage(self, pilImage):
        self.imageQt = ImageQt.ImageQt(pilImage)
        pixmap = QPixmap.fromImage(self.imageQt)
        self.label.setPixmap(pixmap)

    @pyqtSlot()
    def clearImage(self):
        self.label.clear()
        self.imageQt = None


class MaskPreview(QWidget):
    def __init__(self, parent=None, outputSize: Optional[Tuple[int, int]] = None):
        super(MaskPreview, self).__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Raw Image'))
        self.rawImage = ImageView()
        layout.addWidget(self.rawImage)
        layout.addWidget(QLabel('ROI Image'))
        self.roiImage = ImageView()
        layout.addWidget(self.roiImage)
        layout.addWidget(QLabel('Mask Image'))
        self.maskImage = ImageView()
        layout.addWidget(self.maskImage)
        layout.addWidget(QLabel('Blend Image'))
        self.blendImage = ImageView()
        layout.addWidget(self.blendImage)

        self.padLeft = 0
        self.padRight = 0
        self.padTop = 0
        self.padBottom = 0
        self.number = '0'

        self.outputSize = outputSize
        self.roiPilImage = None
        self.masPilImage = None

    @pyqtSlot(str)
    def setPadLeft(self, padLeft: str):
        if padLeft.isnumeric():
            self.padLeft = int(padLeft)
            self._update()

    @pyqtSlot(str)
    def setPadRight(self, padRight: str):
        if padRight.isnumeric():
            self.padRight = int(padRight)
            self._update()

    @pyqtSlot(str)
    def setPadTop(self, padTop: str):
        if padTop.isnumeric():
            self.padTop = int(padTop)
            self._update()

    @pyqtSlot(str)
    def setPadBottom(self, padBottom: str):
        if padBottom.isnumeric():
            self.padBottom = int(padBottom)
            self._update()

    @pyqtSlot(str)
    def setNumber(self, number_text):
        self.number = number_text
        self._update()

    def setImage(self, pilImage):
        self.pilImage = pilImage
        self.rawImage.setImage(self.pilImage)

        ret, self.roiPilImage = crop_roi_by_corners(self.pilImage)
        self._update()

    def _update(self):
        if self.roiPilImage is None:
            self.maskImage.clearImage()
            self.roiImage.clearImage()
            self.blendImage.clearImage()
        else:
            if self.outputSize is not None:
                self.roiPilImage = self.roiPilImage.resize(self.outputSize)
            self.maskPilImage = self._generate_mask()
            mask_orig_size = self.maskPilImage.size
            self.maskPilImage = self.maskPilImage.resize(self.roiPilImage.size)
            self.maskImage.setImage(self.maskPilImage)
            self.roiImage.setImage(self.roiPilImage)
            blend = Image.blend(self.roiPilImage, self.maskPilImage, 0.5)
            blend = blend.resize(mask_orig_size)  # to visualize only
            self.blendImage.setImage(blend)

    def _generate_mask(self):
        num_digits = 7
        if self.number.isnumeric():
            number_text = f'{int(self.number):{num_digits}d}'
        else:
            number_text = self.number
        pad = (self.padLeft, self.padTop, self.padRight, self.padBottom)
        mask = generate(args.font_path, args.font_size, number_text, (0, 0, 0),
                        (255, 255, 255), pad=pad)
        return mask

class App(QMainWindow):
    def __init__(self, imageDir: Path, outputDir: Path, outputSize: Optional[Tuple[int, int]] = None, ext='png'):
        super(App, self).__init__()
        self.outputDir = outputDir
        self.outputDir.mkdir(exist_ok=True, parents=True)
        self.ext = ext
        root = QWidget()
        layout = QHBoxLayout(root)

        self.maskPreview = MaskPreview(root, outputSize)
        layout.addWidget(self.maskPreview)

        self.config = Config(root)
        layout.addWidget(self.config)
        self.setCentralWidget(root)

        self.config.leftPad.connect(self.maskPreview.setPadLeft)
        self.config.rightPad.connect(self.maskPreview.setPadRight)
        self.config.topPad.connect(self.maskPreview.setPadTop)
        self.config.botPad.connect(self.maskPreview.setPadBottom)
        self.config.number.connect(self.maskPreview.setNumber)

        menu = self.menuBar()
        viewMenu = menu.addMenu('View')

        nextAction = QAction('Next', self)
        nextAction.setShortcut(Qt.Key_Down)
        nextAction.triggered.connect(self.nextImage)
        prevAction = QAction('Prev', self)
        prevAction.setShortcut(Qt.Key_Up)
        prevAction.triggered.connect(self.prevImage)
        viewMenu.addAction(nextAction)
        viewMenu.addAction(prevAction)

        self.currentIndex = 0
        self.imagePaths = sorted(list(imageDir.glob(f'*.{ext}')))[:]
        assert len(self.imagePaths) > 0
        self.config.setPad((45, 20, 30, 20))
        self.config.setNumber('')

    def refresh(self):
        self.setImage(self.imagePaths[self.currentIndex])

    def nextImage(self):
        self.save()
        self.currentIndex = min(self.currentIndex + 1, len(self.imagePaths) - 1)
        imagePath = self.imagePaths[self.currentIndex]
        self.setImage(imagePath)

    def prevImage(self):
        self.save()
        self.currentIndex = max(self.currentIndex - 1, 0)
        imagePath = self.imagePaths[self.currentIndex]
        self.setImage(imagePath)

    def setImage(self, imagePath):
        print(f'Processing {imagePath}')
        pilImage = Image.open(imagePath)

        outputConfigPath = self.outputDir.joinpath(imagePath.name).with_suffix('.json')
        if outputConfigPath.exists():
            config = json.load(open(outputConfigPath, 'rt'))
            self.config.setPad((config['leftPad'], config['topPad'], config['rightPad'], config['botPad']))
            self.config.setNumber(config['number'])
            print(config)

        self.maskPreview.setImage(pilImage)

    def save(self):
        if self.maskPreview.roiPilImage is not None:
            imagePath = self.imagePaths[self.currentIndex]
            outputPath = self.outputDir.joinpath(imagePath.stem + f'.{self.ext}')
            outputMaskPath = self.outputDir.joinpath(imagePath.stem + f'_mask.{self.ext}')
            outputConfigPath = self.outputDir.joinpath(imagePath.name).with_suffix('.json')

            self.maskPreview.roiPilImage.save(outputPath)
            self.maskPreview.maskPilImage.save(outputMaskPath)
            to_save = {
                'imagePath': imagePath.name,
                'number': self.maskPreview.number,
                'leftPad': self.maskPreview.padLeft,
                'rightPad': self.maskPreview.padRight,
                'topPad': self.maskPreview.padTop,
                'botPad': self.maskPreview.padBottom,
                'fontSize': 96,  # TODO: hard coded for now
            }
            json.dump(to_save, open(outputConfigPath, 'wt'), indent=4, ensure_ascii=False)
            print(to_save)
            print(f'Saved to {outputPath}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('font_path', type=Path)
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--font-size', default=96, type=int)
    parser.add_argument('--ext', default='png')
    parser.add_argument('--size', type=int, nargs=2, default=None)
    parser.add_argument('--pad', type=int, nargs=4, default=(0, 0, 0, 0))
    args = parser.parse_args()

    # parser.add_argument('font_path', type=Path)
    # parser.add_argument('image_path', type=Path)
    # parser.add_argument('output_dir', type=Path)
    # parser.add_argument('number', type=int)
    # parser.add_argument('--font-size', default=96, type=int)
    # parser.add_argument('--ext', default='png')
    # parser.add_argument('--size', type=int, nargs=2, default=None)
    # parser.add_argument('--pad', type=int, nargs=4, default=(0, 0, 0, 0))
    # args = parser.parse_args()
    print(args)

    app = QApplication([])
    main = App(args.image_dir, args.output_dir, args.size)
    main.refresh()
    main.show()
    app.exec()

    # input_path = Path(args.image_path)
    # input_image = Image.open(input_path)
    # roi_image = crop_roi_by_corners(input_image)
    # mask = generate_mask(roi_image, str(args.number))
    # blend = Image.blend(roi_image, mask, 0.5)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # image = np.array(input_image)

    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # corners = cv2.cornerHarris(gray, 10, 3, 0.24)
    # corners = cv2.dilate(corners, None)
    # ret, corners = cv2.threshold(corners, 0.01*corners.max(), 255, 0)
    # corners = np.uint8(corners)
    # # find centroids
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners)

    # # define the criteria to stop and refine the corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # corners = heuristic_position_corner_filter(corners)
    # assert len(corners) == 4

    # corners = order_points(corners)
    # # for x, y in corners:
    # #     x, y = int(x), int(y)
    # #     cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

    # tl, tr, br, bl = corners
    # width = int(np.round(max([distance(tl, tr), distance(bl, br)])))
    # height = int(np.round(max([distance(tl, bl), distance(tr, br)])))

    # dst = np.array([[0, 0],
    #                 [width - 1, 0],
    #                 [width - 1, height - 1],
    #                 [0, height - 1]], dtype=np.float32)
    # M = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), dst)
    # image = cv2.warpPerspective(image, M, (width, height))

    # input_image = Image.fromarray(image)
    # # input_image.show()

    # # cv2.imshow('Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # # key = cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # # args.output_dir.mkdir(exist_ok=True, parents=True)
    # number = 615
    # num_digits = 7
    # number_text = f'{number:{num_digits}d}'
    # image = generate(args.font_path, args.font_size, number_text, (0, 0, 0), (255, 255, 255), args.size, pad=(45, 20, 30, 20))
    # # output_path = args.output_dir.joinpath(number_text + f'.{args.ext}')
    # # image.save(output_path)

    # # # Test
    # # input_path = '/home/vinhloiit/projects/scale-digits-recognition/data/frames/MAH05484_textline/5000/MAH05484_175.png'
    # # input_image = Image.open(input_path)

    # resized_input = input_image.resize(image.size)
    # # resized_image = image.resize(input_image.size)

    # # # input_image.show()
    # # # resized_image.show()
    # # # image.show()
    # # resized_input.show()
    # Image.blend(image, resized_input, 0.5).show()

    ###########################################################################################
    # input_path = Path(
    #     '/home/vinhloiit/projects/scale-digits-recognition/data/frames/MAH05484_skip10_textline/MAH05484_810.png')
    # input_image = Image.open(input_path)
    # image = np.array(input_image)

    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # corners = cv2.cornerHarris(gray, 10, 3, 0.24)
    # corners = cv2.dilate(corners, None)
    # ret, corners = cv2.threshold(corners, 0.01*corners.max(), 255, 0)
    # corners = np.uint8(corners)
    # # find centroids
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners)

    # # define the criteria to stop and refine the corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    # corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # corners = heuristic_position_corner_filter(corners)
    # assert len(corners) == 4

    # corners = order_points(corners)
    # # for x, y in corners:
    # #     x, y = int(x), int(y)
    # #     cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
    # # cv2.imshow('Corners', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # # key = cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    # tl, tr, br, bl = corners
    # width = int(np.round(max([distance(tl, tr), distance(bl, br)])))
    # height = int(np.round(max([distance(tl, bl), distance(tr, br)])))

    # dst = np.array([[0, 0],
    #                 [width - 1, 0],
    #                 [width - 1, height - 1],
    #                 [0, height - 1]], dtype=np.float32)
    # M = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), dst)
    # image = cv2.warpPerspective(image, M, (width, height))

    # input_image = Image.fromarray(image)
    # # input_image.show()

    # # args.output_dir.mkdir(exist_ok=True, parents=True)
    # number = 3549
    # num_digits = 7
    # number_text = f'{number:{num_digits}d}'
    # image = generate(args.font_path, args.font_size, number_text, (0, 0, 0), (255, 255, 255), args.size, pad=(30, 20, 30, 20))
    # # output_path = args.output_dir.joinpath(number_text + f'.{args.ext}')
    # # image.save(output_path)

    # # # Test
    # # input_path = '/home/vinhloiit/projects/scale-digits-recognition/data/frames/MAH05484_textline/5000/MAH05484_175.png'
    # # input_image = Image.open(input_path)

    # resized_input = input_image.resize(image.size)
    # # resized_image = image.resize(input_image.size)

    # # # input_image.show()
    # # # resized_image.show()
    # image.show()
    # resized_input.show()
    # # Image.blend(image, resized_input, 0.5).show()
