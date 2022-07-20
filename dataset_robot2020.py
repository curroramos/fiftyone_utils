#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 09:04:50 2022

@author: curro
"""

# Use of fiftyone to build a dataset from different dataset folders

import os
import fiftyone as fo
import fiftyone.zoo as foz

#%% aeroHispalis dataset
json_path = "/home/curro/dataset_catec/json_datasets/aerohispalis.json"
images_dir = "/home/curro/dataset_catec/aeroHispalis"

# Load COCO formatted dataset
dataset_aerohispalis = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)

#%% atlas dataset
json_path = "/home/curro/dataset_catec/json_datasets/atlas.json"
images_dir = "/home/curro/dataset_catec/atlas"

# Load COCO formatted dataset
dataset_atlas = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)

#%% beas dataset
json_path = "/home/curro/dataset_catec/json_datasets/beas.json"
images_dir = "/home/curro/dataset_catec/beas"

# Load COCO formatted dataset
dataset_beas = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)

#%% ilipa dataset
json_path = "/home/curro/dataset_catec/json_datasets/ilipa.json"
images_dir = "/home/curro/dataset_catec/ilipa"

# Load COCO formatted dataset
dataset_ilipa = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)


#%% oran dataset
json_path = "/home/curro/dataset_catec/json_datasets/oran.json"
images_dir = "/home/curro/dataset_catec/oran"

# Load COCO formatted dataset
dataset_oran = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)



#%% TRAIN
oran_view = dataset_oran.take(3500)
aerohispalis_view = dataset_aerohispalis.take(3500)

#create train
train = fo.Dataset()
train.add_samples(oran_view)
train.add_samples(aerohispalis_view)

#%% VAL
atlas_view = dataset_atlas.take(1000)

#create valid
valid = fo.Dataset()
valid.add_samples(atlas_view)

#%% TEST
#create test
test = fo.Dataset()
test.add_samples(dataset_beas)
test.add_samples(dataset_ilipa)

#%%
session = fo.launch_app(train)


#%% Export to yolo
EXPORT_DIR = "/home/curro/dataset_catec/robot_2020_v2/yolov5"
train.export(export_dir=EXPORT_DIR, dataset_type=fo.types.YOLOv5Dataset(), split = 'train', classes = ["airplane"])
valid.export(export_dir=EXPORT_DIR, dataset_type=fo.types.YOLOv5Dataset(), split = 'val', classes = ["airplane"])
test.export(export_dir=EXPORT_DIR, dataset_type=fo.types.YOLOv5Dataset(), split = 'test', classes = ["airplane"])


#%% Export to coco
EXPORT_DIR = "/home/curro/dataset_catec/robot_2020_v2/coco/"
#%%
train.export(export_dir=EXPORT_DIR, dataset_type=fo.types.COCODetectionDataset(), labels_path="train.json", classes = ["airplane"], export_media=False)
#%%
valid.export(export_dir=EXPORT_DIR, dataset_type=fo.types.COCODetectionDataset, labels_path="validation.json" , classes = ["airplane"], export_media=False)
#%%
test.export(export_dir=EXPORT_DIR, dataset_type=fo.types.COCODetectionDataset, labels_path="test.json", classes = ["airplane"], export_media=False)

