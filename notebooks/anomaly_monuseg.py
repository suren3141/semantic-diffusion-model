# Import the datamodule
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode
import os

train_dirs = {
    "train": "/mnt/dataset/MoNuSeg/patches_valid_inst_128x128_128x128/__ResNet50_umap_n_components_3_random_state_42_hdbscan_min_samples_10_min_cluster_size_50_v1.2/6/MoNuSegTrainingData/",
}

# Create the datamodule
datamodule = Folder(
    name="monuseg",
    root=train_dirs['train'],
    normal_dir="images",
    task="classification",
    image_size=(128, 128)
)

# Setup the datamodule
datamodule.setup()
