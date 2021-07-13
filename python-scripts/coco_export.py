import json
import os
from shutil import copyfile
from typing import List

import numpy as np
from PIL import Image, ImageFilter, ImageChops
from sklearn.model_selection import train_test_split


# Export the dataset to COCO json format
# Also do a train, valid, test split of the dataset, while respecting the z-stacks / grouped nature of the dataset


# for each folder:
#  1. group all z-stacks, include as arrays
#  2. do a train 0.8, test 0.1, valid 0.1 split
#  3. flatten z-stacks into single arrays
#  4. merge _annotations.json, incorporating the labels, store in memory
#  5. resize images to custom size (optional)
#  6. copy images to new directory with train, test, valid folders, generate COCO JSON files _annotations.coco.json

# constants:

default_width = 1920
default_height = 1080
valid_labels = ["poaceae", "corylus", "alnus"]

# parameters:

train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

image_width = 416
image_height = 416  # round(default_height * (image_width / default_width))


def group_images(images_array: List, groups: List[int]) -> List:
    grouped_images = []
    image_idx = 0
    for group_size in groups:
        grouped_images.append(images_array[image_idx:image_idx + group_size])
        image_idx += group_size
    return grouped_images


def train_test_valid_split(grouped_images, train_size, val_size, test_size, random_state):
    x_train, x_test = train_test_split(grouped_images, test_size=1. - train_size, random_state=random_state)
    x_val, x_test = train_test_split(x_test, test_size=test_size / (test_size + val_size),
                                     random_state=random_state)
    return x_train, x_test, x_val


def resize_image(src, dest, width, height):
    jpeg_image: Image.Image = Image.open(src)
    resized_image = jpeg_image.resize((width, height))
    resized_image.save(dest, **{
        "quality": 98,
        "subsampling": "4:4:4"
    })
    resized_image.close()
    jpeg_image.close()


image_file_path = os.path.abspath(os.path.normpath("../JPEGs"))


def main():
    image_basename_to_full_path = dict()
    train_filenames = dict()
    test_filenames = dict()
    validation_filenames = dict()
    for dataset_folder in os.listdir(image_file_path):
        groups = []
        with open(dataset_folder + "_groups.txt") as f:
            for line in f.readlines():
                groups.append(int(line.strip()))
        dataset_folder_path = os.path.join(image_file_path, dataset_folder)
        image_files = sorted([x for x in os.listdir(dataset_folder_path) if x.endswith(".jpg")])
        for image in image_files:
            image_basename_to_full_path[image] = os.path.join(dataset_folder_path, image)
        grouped_images = group_images(image_files, groups)
        x_train, x_test, x_val = train_test_valid_split(grouped_images, train_size=train_ratio, test_size=test_ratio,
                                                        val_size=validation_ratio, random_state=1)

        def flatten(array):
            return [item for sublist in array for item in sublist]
        x_train = flatten(x_train)
        x_test = flatten(x_test)
        x_val = flatten(x_val)

        cloud_annotations_data = dict()
        with open(os.path.join(dataset_folder_path, "_annotations.json")) as f:
            cloud_annotations_data = json.load(f)

        for images, target_dict in zip([x_train, x_test, x_val],
                                       [train_filenames, test_filenames, validation_filenames]):
            for image in images:
                annotations_data = []
                for annotation in cloud_annotations_data["annotations"][image]:
                    if annotation["label"] in valid_labels:
                        annotations_data.append(annotation)
                target_dict[image] = annotations_data

    # copy images to train, test, valid folders, resize if necessary
    COCO_folder = os.path.abspath(os.path.normpath("../NTNU-pollen-2021-COCO"))
    if image_width != default_width:
        print("Will resize images from", default_width, default_height, "to", image_width, image_height)
    else:
        print("Will not resize images, copying instead")
    for folder_name, image_dict in zip(["train", "test", "valid"],
                                       [train_filenames, test_filenames, validation_filenames]):
        target_directory = os.path.join(COCO_folder, folder_name)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        for image in image_dict:
            image_path = image_basename_to_full_path[image]
            copy_to_path = os.path.join(target_directory, image)
            if image_width != default_width:
                resize_image(image_path, copy_to_path, image_width, image_height)
            else:
                copyfile(image_path, copy_to_path)

        # generate COCO JSON annotations file
        # see: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format

        coco_data = dict()
        coco_data["info"] = {
            "description": "The NTNU-pollen-2021 pollen imagery dataset",
            "url": "https://github.com/Artorp/NTNU-pollen-2021-dataset",
            "version": "1.0",
            "year": 2021,
            "contributor": "Thomas Bruvold",
            "date_created": "2021-07-12"
        }
        coco_data["licenses"] = []
        coco_data["images"] = []
        coco_data["annotations"] = []
        coco_data["categories"] = [
            {
                "id": 0,
                "name": "pollen",
                "supercategory": "none"
            },
            {
                "id": 1,
                "name": "poaceae",
                "supercategory": "pollen"
            },
            {
                "id": 2,
                "name": "corylus",
                "supercategory": "pollen"
            },
            {
                "id": 3,
                "name": "alnus",
                "supercategory": "pollen"
            }
        ]

        image_count = 0
        annotation_count = 0
        label_category = {
            "poaceae": 1,
            "corylus": 2,
            "alnus": 3
        }
        for image in image_dict:
            image_path = os.path.join(target_directory, image)
            annotations = image_dict[image]
            coco_data["images"].append({
                "file_name": image,
                "height": image_height,
                "width": image_width,
                "id": image_count
            })
            for annotation in annotations:
                x = annotation["x"] * image_width
                y = annotation["y"] * image_height
                x2 = annotation["x2"] * image_width
                y2 = annotation["y2"] * image_height
                box_width = x2 - x
                box_height = y2 - y
                coco_data["annotations"].append({
                    "id": annotation_count,
                    "image_id": image_count,
                    "category_id": label_category[annotation["label"]],
                    "bbox": [
                        int(x), int(y),
                        int(box_width), int(box_height)
                    ],
                    "area": box_width * box_height,
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotation_count += 1
            image_count += 1

        with open(os.path.join(target_directory, "_annotations.coco.json"), "w") as f:
            json.dump(coco_data, f)


if __name__ == '__main__':
    main()
