#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 12:59:34 2022

@author: curro
"""

import os
import fiftyone as fo
import fiftyone.zoo as foz


#%% oran dataset
json_path = "/home/curro/mmdetection/data/coco_data/train.json"
images_dir = "/home/curro/mmdetection/data/coco_data/coco_train"

# Load COCO formatted dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)

view = dataset.shuffle()

session = fo.launch_app(view)

session.wait()