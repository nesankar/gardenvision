from pathlib import Path
from plantcv import plantcv as pcv
import cv2
import glob
from typing import Union, List
import sys

from src.plant_image import PlantImage


def load_dir_images(
    directory: Union[str, Path], file_type: str, tool: str = "plantcv", ref_length=2.0
) -> List[PlantImage]:
    """Given a directory and a file type load all images inside"""

    # Get all files of file_type in the dir...
    files = glob.glob(str(directory / f"*.{file_type}"))

    # ... and then load and return the images as a list.
    if tool == "plantcv":
        return [
            PlantImage(
                pcv.readimage(file)[0],
                file.split("/")[-1].split(".")[0],
                reference_obj_max_dim_in=ref_length,
            )
            for file in files
        ]
    else:
        return [
            PlantImage(cv2.imread(file), files.split("/")[-1].split(".")[0])
            for file in files
        ]
