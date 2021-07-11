import os
from PIL import Image, ImageFilter, ImageChops
import numpy as np
from matplotlib import pyplot as plt
import skimage.filters

image_file_path = os.path.abspath(os.path.normpath("../JPEGs"))

# The beginning of an autogrouping algorithm, used to


def main():
    for dataset_folder in os.listdir(image_file_path):  # kjevik_2020-06-25
        dataset_folder_path = os.path.join(image_file_path, dataset_folder)

        # ['Snap-235.jpg', 'Snap-236.jpg', ...
        image_files = sorted([x for x in os.listdir(dataset_folder_path) if x.endswith(".jpg")])
        image_files = image_files[-80:]  # TODO: for testing

        mean_differences = []

        # calculate mean differences
        for image_name_a, image_name_b in zip(image_files, image_files[1:]):
            image_a: Image.Image = Image.open(os.path.join(dataset_folder_path, image_name_a))
            image_b: Image.Image = Image.open(os.path.join(dataset_folder_path, image_name_b))
            image_a = image_a.filter(ImageFilter.GaussianBlur(20))
            image_b = image_b.filter(ImageFilter.GaussianBlur(20))
            # print(image_name_a, image_name_b)
            # image_a.save("a.jpg")
            # image_b.save("b.jpg")
            image_diff = ImageChops.difference(image_a, image_b)
            # image_diff.save("diff.jpg")
            image_diff_arr = np.array(image_diff, dtype="float") / 255
            mean_diff = image_diff_arr.mean()
            mean_differences.append(mean_diff)
            print(image_name_a, "=>", image_name_b, "mean difference is", mean_diff)
            # TODO: use Otsu's method to find the threshold that minimized intra-class variance
            #   note: minimizing intra-class variance is the same as maximizing inter-class variance
            if mean_diff > 0.01:
                print("I think I detected a grouping difference")

        # plot color differences
        plt.hist(mean_differences, bins=100)
        plt.show()
        print("\n========================\n")

        # use Otsu's method to find a good guess of threshold
        threshold = skimage.filters.threshold_otsu(np.array(mean_differences), nbins=100)
        print("Found optimal threshold with Otsu's method:", threshold)

        grouped_images = []
        working_group = []

        for i, (image_name_a, image_name_b) in enumerate(zip(image_files, image_files[1:])):
            mean_diff = mean_differences[i]
            print(image_name_a, "=>", image_name_b, "mean difference is", mean_diff)
            working_group.append(image_name_a)
            if mean_diff > threshold:
                print("I think I detected a grouping difference")
                grouped_images.append(working_group)
                working_group = []
        grouped_images.append(working_group)

        print(grouped_images)
        for group in grouped_images:
            print(group)
        break

    return
    image_files = sorted([x for x in os.listdir(image_file_path) if x.endswith(".jpg")])
    print(image_files)
    for image_name_a, image_name_b in zip(image_files, image_files[1:]):
        image_a: Image.Image = Image.open(os.path.join(image_file_path, image_name_a))
        image_b: Image.Image = Image.open(os.path.join(image_file_path, image_name_b))
        image_a = image_a.resize((480, 270)).filter(ImageFilter.GaussianBlur(10))
        image_b = image_b.resize((480, 270)).filter(ImageFilter.GaussianBlur(10))
        # print(image_name_a, image_name_b)
        # image_a.save("a.jpg")
        # image_b.save("b.jpg")
        image_diff = ImageChops.difference(image_a, image_b)
        # image_diff.save("diff.jpg")
        image_diff_arr = np.array(image_diff, dtype="float")/255
        mean_diff = image_diff_arr.mean()
        print(image_name_a, "=>", image_name_b, "mean difference is", mean_diff)
        # TODO: use Otsu's method to find the threshold that minimized intra-class variance
        #   note: minimizing intra-class variance is the same as maximizing inter-class variance
        if mean_diff > 0.01:
            print("I think I detected a grouping difference")
        # BTW: point matching algorithm:
        #   1. select a random point a (that is not on an edge). Find 5 nearest neighbors, b_1, .., b_5
        #   2. for each match, define xy-offset transformation a -> b_1: o_1, .., o_5
        #   3. for each offset o_i:
        #      a. transform all points by o_i
        #      b. calculate nearest neighbor for all points.
        #      c. calculate square distance between all points. sum it.
        #   4. the offset transformation that minimizes the sum square error is (probably) the correct offset
        #   5. Note: After the correct offset is found, can use the pairings to calculate best offset which
        #      minimizes the squared distance. That way, when using IoU later on, it's possible to calculate
        #      a better IoU on the transformed bounding box
        #   6. PPS: use the first image as the anchor, and offset all other images to the first image?
        # Alternatively: treat the problem as a balanced assignment problem and use the Hungarian algorithm
        #     will probably not work as the cost is a function of all points, which degrades computational complexity
        # Alternatively: select two points at random that have some distance between each other
        #     then, find 3 nearest neighbors for each point, and for the 3*3 possible combinations,
        #     calculate all affine transformations between those two points. Transform all points by those,
        #     then calculate the squared distance. Pros: will support scale and rotation. Cons: more computationally
        #     complex, and scaling and rotation might not be needed


if __name__ == '__main__':
    main()
