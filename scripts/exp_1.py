import sys
from pathlib import Path

"""Simple script to explore loading an image"""

file_dir = Path(__file__)
root = file_dir.parent.parent
sys.path.append(root)

from src import basic_utils


if __name__ == "__main__":
    img_path = root / "resources" / "misc_data" / "first_cut"
    img_type = "jpeg"
    images = basic_utils.load_dir_images(img_path, img_type)

    print(
        f"""\nThe size of the images is {" x ".join([str(dim) for dim in images[0].shape])} pixels."""
    )
