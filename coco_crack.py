
#%%
import fiftyone as fo 

json_path = "/home/curro/dataset_catec/crack_detection_dataset/train.json"
images_dir = "/home/curro/dataset_catec/crack_detection_dataset/train"

#%% Load COCO formatted dataset
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=images_dir,
    labels_path=json_path,
    include_id=True,
    label_field="labels",
)

#%% Get labeled images and non_labeled
from fiftyone import ViewField as F

labeled_view = dataset.filter_labels("labels_detections",(F("label") == "fissure") )
non_labeled_view  = dataset.exclude(labeled_view)


#%% TRAIN
labeled_selected_view = labeled_view.take(len(non_labeled_view))

#create train
train = fo.Dataset()
train.add_samples(non_labeled_view)
train.add_samples(labeled_selected_view)


#%% Run fiftyone
view = train.shuffle()
session = fo.launch_app(view)

session.wait()

#%% Exportar a image classification task. Export to directory tree
EXPORT_DIR = "/home/curro/dataset_catec/crack_detection_dataset/image_classification/normal"
non_labeled_view.export(export_dir=EXPORT_DIR, dataset_type=fo.types.ImageDirectory())

EXPORT_DIR = "/home/curro/dataset_catec/crack_detection_dataset/image_classification/fissure"
labeled_selected_view.export(export_dir=EXPORT_DIR, dataset_type=fo.types.ImageDirectory())
