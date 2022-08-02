# fiftyone_utils

This repo contains sample files that can be used to manipulate and visualize Object Detection image datasets.

You can find fiftyone documentation [here](https://voxel51.com/docs/fiftyone/ "Fiftyone documentation").

The operations explained in this repo include the following:
- Load a dataset
- Separate the images with objects from non-object images
- Create a new dataset images from different datasets
- Export the created dataset to different formats


## 0. Imports
To use fiftyone library, just import the following in a python script:
```
import fiftyone as fo
```

## 1. Load a dataset
### 1.1 Load a COCO formatted dataset

If we have a COCO dataset:

```
# Dataset directory
json_path = <json_path> # Path to json file containing the images labels 
images_dir = <image_dir>  # Root folder with the images

dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset, 
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)
```

### 1.2 Load a YOLO formatted dataset

Otherwise, if we have the labels in YOLO format (.txt), we can load the dataset specifying the root folder and the `.yaml`:

```
# Dataset directory
dataset_dir = "<data path>"
yaml_path = "<path-to-dataset.yml>"

# Load YOLO formatted dataset
yolo_dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.YOLOv5Dataset(),
    dataset_dir=dataset_dir,
    yaml_path = yaml_path,
    include_id=True,
    label_field="",
)
```

### 1.3 Change YOLO format to COCO format
Other option is to use the repo: https://github.com/Taeyoung96/Yolo-to-COCO-format-converter to change the format.

Follow these steps:
1. Install the repo
2. Make sure to have only one folder including the images with their respective labels (the differences in their name is the extension jpg-txt, png-txt, etc-txt)
3. Modify `main.py` with the correct class names, e.g.
```
classes = [
    "airplane",
    # "handle",
    # "table",
    # "button",
    # "person",
]

```
4. Run the following command:
```
python3 main.py -p $FOLDER_PATH$ --output $OUTPUT_FILENAME$.json
```

## 2. Create a new dataset from different datasets
1. Load the datasets
2. Create the new dataset
```
dataset = fo.Dataset()
```
3. Take the desired number of samples from each dataset
```
num_samples = 4000

# Take samples
oran_view = dataset_oran.take(num_samples)
beas_view = dataset_beas.take(num_samples)
ilipa_view = dataset_ilipa.take(num_samples)
```

4. Add the selected samples to the new dataset
```
# Add samples
dataset.add_samples(oran_view)
dataset.add_samples(beas_view)
dataset.add_samples(ilipa_view)
```


## 3. Filter labelled and unlabelled data
(When we talk about unlabelled data, we refer to images where no target objects are present)

Steps:
1. Obtain labelled data
```
from fiftyone import ViewField as F

class_name = "airplane" # Wanted class

labeled_view = dataset.filter_labels("labels_detections",(F("label") == class_name) )
```

2. Exclude labelled data from whole dataset to obtain unlabelled data
```
non_labeled_view  = dataset.exclude(labeled_view)
```

3. Use these 2 different distributions to create a new balanced dataset
```
# Create new dataset
dataset = fo.Dataset()

# Take the same number of labelled and unlabelled samples
labeled_reduced_view = labeled_view.take(len(non_labeled_view))

# Add samples to the new dataset 
dataset.add_samples(labeled_reduced_view)
dataset.add_samples(non_labeled_view)
```
## 4. Create train, validation and test sets
To create train, validation and test sets we take the desired percentage of images from the desired distributions (scenarios). 
- train-val => 70%-30% from same distribution
- test => length==val using images comming from a different distribution (different scenarios) to make sure our model has been able to generalize.
For example:
```
# Train set
oran_view = dataset_oran.take(3500)
aerohispalis_view = dataset_aerohispalis.take(3500)
 
train = fo.Dataset()
train.add_samples(oran_view)
train.add_samples(aerohispalis_view)

# Validation set
atlas_view = dataset_atlas.take(1000)

valid = fo.Dataset()
valid.add_samples(atlas_view)

# Test set
test = fo.Dataset()
test.add_samples(dataset_beas)
test.add_samples(dataset_ilipa)
```

## 5. Export new dataset to different formats
Once we have manipulated and created our new dataset, we can export it into different formats
# 5.1 Export to COCO
Export resulting `.json` files (If you also want to export the images files change `export_media=True`)
```
EXPORT_DIR = "<root-out-dataset-dir>"
dataset_classes = ["airplane"]

train.export(export_dir=EXPORT_DIR, dataset_type=fo.types.COCODetectionDataset, labels_path="train.json", classes = dataset_classes, export_media=False)
valid.export(export_dir=EXPORT_DIR, dataset_type=fo.types.COCODetectionDataset, labels_path="validation.json" , classes = dataset_classes, export_media=False)
test.export(export_dir=EXPORT_DIR, dataset_type=fo.types.COCODetectionDataset, labels_path="test.json", classes = dataset_classes, export_media=False)
```

# 5.2 Export to YOLO
```
EXPORT_DIR = "<root-out-dataset-dir>"
dataset_classes = ["airplane"]

train.export(export_dir=EXPORT_DIR, dataset_type=fo.types.YOLOv5Dataset(), split = 'train', classes = dataset_classes)
valid.export(export_dir=EXPORT_DIR, dataset_type=fo.types.YOLOv5Dataset(), split = 'val', classes = dataset_classes)
test.export(export_dir=EXPORT_DIR, dataset_type=fo.types.YOLOv5Dataset(), split = 'test', classes = dataset_classes)
```

# 5.3 Export dataset for Image Classification task
For Image Classification tasks we need the dataset to be distributed in a directory tree in which different classes are separed. Supposing we want to classify if there is an object present or not (regardless of their position), we use the following code.

```
# First folder (no objects in images)
EXPORT_DIR = "/home/curro/datasets/crack_dataset/image_classification/evaluation/normal"
non_labeled_view.export(export_dir=EXPORT_DIR, dataset_type=fo.types.ImageDirectory())

# Second folder (present objects in images)
EXPORT_DIR = "/home/curro/datasets/crack_dataset/image_classification/evaluation/fissure"
labeled_selected_view.export(export_dir=EXPORT_DIR, dataset_type=fo.types.ImageDirectory())
```