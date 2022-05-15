import sys
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

"""Simple script to explore loading an image and some analyses"""

file_dir = Path(__file__)
root = file_dir.parent.parent
sys.path.append(str(root))

from src import basic_utils


if __name__ == "__main__":
    img_path = root / "resources" / "misc_data" / "first_batch"
    img_type = "jpeg"
    images = basic_utils.load_dir_images(img_path, img_type)

    print(
        f"""\nThe size of the images is {" x ".join([str(dim) for dim in images[0].image.shape])} pixels."""
    )

    # extract the plants for each image
    plants = [basic_utils.do_segmentation(image.image, dark_background=False) for image in tqdm(images)]

    # plot the colors of the plants in each image
    for i, plant in enumerate(plants):
        basic_utils.do_color_analysis(plant, plot=True, plot_title=images[i].name)


    bp=1
