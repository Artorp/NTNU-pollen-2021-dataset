import os
from PIL import Image, ImageDraw, ImageFont

# Each group is one z-stack from the same position with different
# camera zoom. The groupings were manually created into *_groups.txt files, line separated
# with the number of images in a group.

# validate total number of images, and create temp image folder with group text for validation

# This script should only be used to validate the manually created _groups.txt files


jpeg_compression_options = {
    "quality": 95,
    "subsampling": "4:4:4"
}


image_file_path = os.path.abspath(os.path.normpath("../JPEGs"))


def main():
    for dataset_folder in os.listdir(image_file_path):  # e.g. kjevik_2020-06-25
        groups_images_count = 0
        print(dataset_folder)
        with open(dataset_folder + "_groups.txt") as f:
            for line in f.readlines():
                groups_images_count += int(line)
        dataset_folder_path = os.path.join(image_file_path, dataset_folder)
        image_files = sorted([x for x in os.listdir(dataset_folder_path) if x.endswith(".jpg")])
        if len(image_files) != groups_images_count:
            print(f"WARNING: {dataset_folder} group mismatch, {len(image_files)} files and {groups_images_count} total images in groups")

    # create temp folder with text on
    font = ImageFont.truetype("fonts/arialbd.ttf", 14)
    for dataset_folder in os.listdir(image_file_path):
        dataset_folder_path = os.path.join(image_file_path, dataset_folder)
        image_files = sorted([x for x in os.listdir(dataset_folder_path) if x.endswith(".jpg")])
        group_idx = 1
        image_idx = 0
        with open(dataset_folder + "_groups.txt") as f:
            for line in f.readlines():
                num = int(line)
                for i in range(num):
                    current_image_fname_path = os.path.join(dataset_folder_path, image_files[image_idx])
                    jpeg_image: Image.Image = Image.open(current_image_fname_path)
                    draw = ImageDraw.Draw(jpeg_image)
                    draw.text((4, 4), "g_idx: " + str(group_idx), fill="black", font=font, stroke_width=2, stroke_fill="white")
                    jpeg_image.save(current_image_fname_path.replace("JPEGs", "JPEGs_grouped"), **jpeg_compression_options)
                    jpeg_image.close()
                    image_idx += 1
                group_idx += 1



if __name__ == '__main__':
    main()
