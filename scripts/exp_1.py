import sys
from pathlib import Path
from tqdm import tqdm
import concurrent.futures  # TODO: processing is slow, do it in parallel

"""Simple script to explore loading an image and some analyses"""

file_dir = Path(__file__)
root = file_dir.parent.parent
sys.path.append(str(root))

from src import basic_utils, image_io


if __name__ == "__main__":
    img_path = root / "resources" / "misc_data" / "second_reference"
    img_type = "jpeg"
    images = image_io.load_dir_images(img_path, img_type)

    print(
        f"""\nThe size of the images is {" x ".join([str(dim) for dim in images[0].image.shape])} pixels."""
    )

    # extract the plants for each image
    for image in images:
        # do some simple tooling
        image.plot_analyzed_plant()
        print("\n")
        print(f"The total area of the {image.name} plant is {round(image.plant_area_in2, 2)}.")
        print(f"The {image.name} plant is {round(image.plant_length_in, 2)} inches by {round(image.plant_width_in, 2)} inches @ max.")
        print("\n")
        image.get_plant_color_spectrum(plot=True)
