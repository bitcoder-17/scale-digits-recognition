from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.objects import FieldInfo


class DigitRecognizer(object):
    def __init__(self, mode: str):
        checkpoint = torch.load('./classifier_best.pth', map_location='cuda')
        self.class2int = checkpoint['class2int']
        self.int2class = {v: k for k, v in self.class2int.items()}
        self.classifier = Classifier(20, 70, 11).to('cuda')
        self.classifier.load_state_dict(checkpoint['model'])
        self.classifier.eval()

    @torch.no_grad()
    def run(self, field_info: FieldInfo) -> FieldInfo:
        if len(field_info.boxes) == 0:
            return field_info

        crops = [field_info.mask_feature[0, 0, box.points[0].y:box.points[1].y, box.points[0].x:box.points[1].x] for box in field_info.boxes]
        crops = torch.stack(crops, dim=0)
        pred_classes, prob = self.classifier(crops)
        pred_class_str = [self.int2class[pred_] for pred_ in pred_classes.tolist()]
        for pred_class, box in zip(pred_class_str, field_info.boxes):
            box.text = pred_class
            # TODO: Prob here
        return field_info


class Classifier(nn.Module):
    def __init__(self, width, height, num_classes):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten(-2, -1)
        self.linear = nn.Linear(in_features=width*height, out_features=num_classes)

    def forward(self, boxes: torch.Tensor, targets: Optional[torch.Tensor] = None):
        '''
        boxes: [B, H, W]
        targets: [B]
        '''
        boxes = self.flatten(boxes)  # [B, H * W]
        boxes = self.linear(boxes)   # [B, num_classes]
        if self.training:
            assert targets is not None
            loss = F.cross_entropy(boxes, targets, reduction='mean')
            return loss
        else:
            boxes = F.softmax(boxes, dim=-1)        # [B, num_classes]
            probs, argmax = boxes.max(dim=-1)      # [B], [B]
            return argmax, probs
            