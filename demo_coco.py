"""
Sample file 
"""

import fiftyone as fo

# Oran dataset
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

# Create a view with the dataset shuffled
view = dataset.shuffle()

# Open a new session
session = fo.launch_app(view)

# Wait till the session is closed
session.wait()