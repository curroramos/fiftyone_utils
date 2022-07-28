#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 12:11:43 2022

@author: curro
"""

import os

import fiftyone as fo
import fiftyone.zoo as foz

# Load COCO formatted dataset
coco_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.YOLOv5Dataset(),
    dataset_dir="/home/curro/yolo_vigia/runs/coco2yolov5/exp/train",
    #yaml_path = "/home/curro/yolo_vigia/runs/coco2yolov5/exp/dataset.yml",
    include_id=True,
    label_field="",
)
session = fo.launch_app(coco_dataset)

