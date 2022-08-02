
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Other sample file
"""

# Use of fiftyone to build a dataset from different dataset folders
import fiftyone as fo


#%% 20210205 dataset
json_path = "/home/curro/dataset_catec/json_datasets/no_etiquetados.json"
images_dir = "/home/curro/dataset_catec/etiquetados"

# Load COCO formatted dataset
dataset_202102 = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)

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

#%% coger no etiquetados de 202102...

from fiftyone import ViewField as F

labeled_view = dataset_202102.filter_labels("labels_detections",(F("label") == "airplane") )
non_labeled_view  = dataset_202102.exclude(labeled_view)

#%% coger un porcentaje igualitario de cada dataset
num_samples = 4000

oran_view = dataset_oran.take(num_samples)
beas_view = dataset_beas.take(len(dataset_beas))
ilipa_view = dataset_ilipa.take(len(dataset_ilipa))
atlas_view = dataset_atlas.take(len(dataset_atlas))
aerohipalis_view = dataset_aerohispalis.take(num_samples)

#%% create the dataset
dataset = fo.Dataset()
dataset.add_samples(oran_view)
dataset.add_samples(beas_view)
dataset.add_samples(ilipa_view)
dataset.add_samples(atlas_view)
dataset.add_samples(aerohipalis_view)
dataset.add_samples(non_labeled_view)

view = dataset.shuffle()


#%% export dataset as yolo dataset
# EXPORT_DIR = "/home/curro/dataset_catec/result"
# view.export(export_dir=EXPORT_DIR, dataset_type=fo.types.YOLOv5Dataset())

#%% launch fiftyone
session = fo.launch_app(dataset)

#%% split train and validation
train_view = dataset.take(int(len(view)*0.8))
test_view = dataset.exclude(train_view)

#%%
EXPORT_DIR = "/home/curro/dataset_catec/data"
train_view.export(export_dir=EXPORT_DIR, dataset_type=fo.types.YOLOv5Dataset(), split = 'train', classes = ["airplane"])
test_view.export(export_dir=EXPORT_DIR, dataset_type=fo.types.YOLOv5Dataset(), split = 'val', classes = ["airplane"])


#%% OPTIONAL: File with dataset characteristics
import csv
csv_file =  open("dataset_info.csv", 'w')
csv_writer = csv.writer(csv_file, delimiter= '\n')
csv_writer.writerow(['Amount of images from each dataset:'])

model_names = ['Oran', 'Beas', 'Ilipa', 'Atlas', 'Aerohispalis', 'Etiquetados']
model_views = [oran_view,beas_view,ilipa_view,atlas_view,aerohipalis_view,non_labeled_view]

i=0
for model_name in model_names:
    csv_writer.writerow([model_name + ' images:' + str(len(model_views[i]))])
    i+=1    
    
labeled_view_dataset = dataset.filter_labels("labels_detections",(F("label") == "airplane") )
non_labeled_view_dataset  = dataset.exclude(labeled_view_dataset)

csv_writer.writerow(['Summary:'])
csv_writer.writerow(['Labeled images:' + str(len(labeled_view_dataset))])
csv_writer.writerow(['Non-labeled images:' + str(len(non_labeled_view_dataset))])
csv_writer.writerow(['Total images:' + str(len(dataset))])

csv_writer.writerow(['----------'])

csv_writer.writerow(['Split train and valid:'])
csv_writer.writerow(['Train images:' + str(len(train_view))])
csv_writer.writerow(['Valid images:' + str(len(test_view))])


csv_file.close()



