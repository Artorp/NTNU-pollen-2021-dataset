import json
import os
from shutil import copyfile
from typing import List

import numpy as np
from PIL import Image, ImageFilter, ImageChops
from sklearn.model_selection import train_test_split


# Export the dataset to YOLO format, with file paths relative to the darknet executable
# Also do a train, valid, test split of the dataset, while respecting the z-stacks / grouped nature of the dataset


# for each folder:
#  1. group all z-stacks, include as arrays
#  2. do a train 0.8, test 0.1, valid 0.1 split
#  3. flatten z-stacks into single arrays
#  4. merge _annotations.json, incorporating the labels, store in memory
#  5. resize images to custom size (optional)
#  6. copy images to new directory with train, test folders, generate YOLO text files and train.txt & test.txt

# constants:

default_width = 1920
default_height = 1080
valid_labels = ["poaceae", "corylus", "alnus"]

# parameters:

train_ratio = 0.8
test_ratio = 0.2

image_width = 416
image_height = 416  # round(default_height * (image_width / default_width))


def group_images(images_array: List, groups: List[int]) -> List:
    grouped_images = []
    image_idx = 0
    for group_size in groups:
        grouped_images.append(images_array[image_idx:image_idx + group_size])
        image_idx += group_size
    return grouped_images


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
        x_train, x_test = train_test_split(grouped_images, test_size=test_ratio, random_state=1)

        def flatten(array):
            return [item for sublist in array for item in sublist]
        x_train = flatten(x_train)
        x_test = flatten(x_test)

        cloud_annotations_data = dict()
        with open(os.path.join(dataset_folder_path, "_annotations.json")) as f:
            cloud_annotations_data = json.load(f)

        for images, target_dict in zip([x_train, x_test],
                                       [train_filenames, test_filenames]):
            for image in images:
                annotations_data = []
                for annotation in cloud_annotations_data["annotations"][image]:
                    if annotation["label"] in valid_labels:
                        annotations_data.append(annotation)
                target_dict[image] = annotations_data

    # copy images to yolo folder, resize if necessary
    YOLO_folder = os.path.abspath(os.path.normpath("../NTNU-pollen-2021-YOLO-darknet"))
    if image_width != default_width:
        print("Will resize images from", default_width, default_height, "to", image_width, image_height)
    else:
        print("Will not resize images, copying instead")
    for folder_name, image_dict in zip(["train", "test"],
                                       [train_filenames, test_filenames]):
        target_directory = os.path.join(YOLO_folder, folder_name)
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        for image in image_dict:
            image_path = image_basename_to_full_path[image]
            copy_to_path = os.path.join(target_directory, image)
            if image_width != default_width:
                resize_image(image_path, copy_to_path, image_width, image_height)
            else:
                copyfile(image_path, copy_to_path)

        # generate YOLO annotations files

        # train.txt, test.txt
        txt_list_of_images_fn = os.path.join(YOLO_folder, folder_name + ".txt")
        with open(txt_list_of_images_fn, "w") as f:
            for image in image_dict:
                f.write(f"data/{folder_name}/{image}\n")

        image_count = 0
        annotation_count = 0
        label_category = {
            "poaceae": 0,
            "corylus": 1,
            "alnus": 2
        }
        for image in image_dict:
            image_path = os.path.join(target_directory, image)
            basename, ext = os.path.splitext(image)
            annotations = image_dict[image]
            annotations_yolo_lines = []  # class index, x_center, y_center, width, height in relative 0..1 scale
            for annotation in annotations:
                x = annotation["x"]
                y = annotation["y"]
                x2 = annotation["x2"]
                y2 = annotation["y2"]
                box_width = x2 - x
                box_height = y2 - y
                x_center = x + box_width / 2
                y_center = y + box_height / 2
                image_class = label_category[annotation["label"]]
                annotations_yolo_lines.append(f"{image_class} {x_center} {y_center} {box_width} {box_height}\n")

            yolo_text_path = os.path.join(target_directory, basename + ".txt")
            with open(yolo_text_path, "w") as f:
                for line in annotations_yolo_lines:
                    f.write(line)


if __name__ == '__main__':
    main()
