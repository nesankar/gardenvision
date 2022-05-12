import cv2
import glob
import sys
from typing import Union, List, Any
from pathlib import Path

"""Simple processing utilities"""

file_dir = Path(__file__)
root = file_dir.parent.parent
sys.path.append(root)


def load_dir_images(directory: Union[str, Path], file_type: str) -> List[Any]:
    """Given a directory and a file type load all images inside"""

    # Get all files of file_type in the dir...
    files = glob.glob(str(directory / f"*.{file_type}"))

    # ... and then load and return the images as a list.
    return [cv2.imread(file) for file in files]
