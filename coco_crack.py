
import fiftyone as fo 

json_path = "/home/curro/dataset_catec/crack_detection_dataset/annotations.json"
images_dir = "/home/curro/dataset_catec/crack_detection_dataset/train"

#%% Load COCO formatted dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)

#%%
view = dataset.shuffle()

session = fo.launch_app(view)

session.wait()
