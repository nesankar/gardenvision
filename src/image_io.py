from pathlib import Path
from plantcv import plantcv as pcv
import cv2
import glob
from typing import Union, List
import yaml
import logging
from src.plant_image import PlantImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


"""Do the work to read in a directory of plant files."""


def load_dir_images(
    directory: Union[str, Path], file_type: str, tool: str = "plantcv", ref_length=2.0
) -> List[PlantImage]:
    """Given a directory and a file type load all images inside"""

    # Get all files of file_type in the dir...
    files = glob.glob(str(directory / f"*.{file_type}"))

    logger.info(f"Found {len(files)} .{file_type} files in the {directory} folder.")

    # check if there is a metadata file
    if (directory / "metadata.yml").is_file():
        # read it
        with open(directory / "metadata.yml", "r") as md_file:
            dir_metadata = yaml.safe_load(md_file)
        logger.info(f"For the {directory} dir, the metadata is: {dir_metadata}.")
    else:
        # Otherwise error out, b/c there will be very limited functionality.
        raise FileNotFoundError(
            "No metadata.yml file found with the images. Add a metadata file with the 'refrence_obj_color' and 'ref_obj_max_length' data."
        )

    # ... and then load and return the images as a list.
    if tool == "plantcv":
        return [
            PlantImage(
                pcv.readimage(file)[0],
                file.split("/")[-1].split(".")[0],
                ref_obj_color=dir_metadata["reference_obj_color"],
                reference_obj_max_dim_in=dir_metadata["reference_obj_max_length"],
            )
            for file in files
        ]
    else:
        return [
            PlantImage(
                cv2.imread(file),
                files.split("/")[-1].split(".")[0],
                ref_obj_color="yellow",
                reference_obj_max_dim_in=ref_length,
            )
            for file in files
        ]
