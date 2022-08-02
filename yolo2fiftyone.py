"""
Sample file to load a YOLO formatted dataset
"""

import fiftyone as fo

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

# Open a new session
session = fo.launch_app(yolo_dataset)

# wait till the session is closed
session.wait()
