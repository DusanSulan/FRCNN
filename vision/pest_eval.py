
import json
import tempfile

import numpy as np
import copy
import time
import torch
import torch._six

from collections import defaultdict
import utilss as utils

class PestEvaluator(object):
    def __init__(self, dataloader, predictions):
        self.img_ids = []
        
        #predictions = res = labels and boxes, classes , scores
        img_ids = list(np.unique(list(predictions.keys())))
        
        #write predictions
        results = self.prepare_for_coco_detection(predictions)
        #evaluace?
        coco_eval = self.coco_eval

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for coco_eval in self.coco_eval.items():
            coco_eval.summarize()
            
    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "score": scores[k],
                    }
                    for k, box in enumerate(scores)
                ]
            )
        return coco_results


