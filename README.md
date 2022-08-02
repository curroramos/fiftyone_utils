# fiftyone_utils

This repo contains sample files that can be used to manipulate and visualize image datasets.

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
3. Run the following command:
```
python3 main.py -p $FOLDER_PATH$ --output $OUTPUT_FILENAME$.json
```

## 2. 

