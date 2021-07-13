# NTNU-pollen-2021-dataset

This repository contains the NTNU-pollen-2021 dataset, containing microscoping imaging of pollen samples.

They were collected as part of NTNU master theses for spring 2021.

## Usage

The dataset is available in the YOLO and COCO standard formats, with images rescaled to 416x416 px. The COCO dataset is also aplit into a validation set.

Other sizes and the ratio of the train / test split can be adjusted in `coco_export.py` and `yolo_darknet_export.py`.

## Original images

The original pictures are available in 1920x1080 px in the `JPEGs` folder, in the format as they were taken of 400x magnification. The folders are structured by sample slides, as the original images are taken from three seperate slides. The folders also include unofficial `_annotation.json` files for annotating, exported from the cloud.annotations.ai service used to annotate the images.

## Original raw images

The unstructured raw images (as TIF) as well as some metadata and augmentation pipeline are available at:

https://github.com/Artorp/NTNU-Pollen-2021-raws
