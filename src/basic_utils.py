import cv2
import glob
import sys
from typing import Union, List, Any, Optional
from collections import namedtuple
from pathlib import Path
import numpy as np
import pandas_bokeh as pbk
from plantcv import plantcv as pcv
import matplotlib

matplotlib.use("Agg")

"""Simple processing utilities"""

file_dir = Path(__file__)
root = file_dir.parent.parent
sys.path.append(root)

PlantImage = namedtuple("PlantImage", ["image", "name"])
BoxCords = namedtuple("BoxCords", ["ul_corner", "ur_corner", "ll_corner", "lr_corner"])
PlantFeature = namedtuple(
    "PlantFeature", ["plant_obj", "bg_mask", "rgb_image", "analyzed_image"]
)


def load_dir_images(
    directory: Union[str, Path], file_type: str, tool: str = "plantcv"
) -> List[PlantImage]:
    """Given a directory and a file type load all images inside"""

    # Get all files of file_type in the dir...
    files = glob.glob(str(directory / f"*.{file_type}"))

    # ... and then load and return the images as a list.
    if tool == "plantcv":
        return [
            PlantImage(pcv.readimage(file)[0], file.split("/")[-1].split(".")[0])
            for file in files
        ]
    else:
        return [
            PlantImage(cv2.imread(file), files.split("/")[-1].split(".")[0])
            for file in files
        ]


def create_central_bounding_box(
    image: np.ndarray, pct_box_size: float = 0.125
) -> BoxCords:
    """
    # Define the region of interest (ROI) as a  localized central box
    image: the input image as a 3-d numpy array
    pct_box_size: the total fraction of image area that the bounding box should cover


    :return:  NamedTuple of the bounding box corner cordinates
    """

    # First, get the image dimensions...
    length, width, n_colors = image.shape

    # ... compute the midpoints of the respective locations...
    center_width = width // 2
    center_length = length // 2

    # ... use that to define the width and lenght bounding points...
    width_cords = center_width - (int(center_width * pct_box_size)), center_width + (
        int(center_width * pct_box_size)
    )
    length_cords = center_length - (
        int(center_length * pct_box_size)
    ), center_length + (int(center_length * pct_box_size))

    # ... and return the resulting named tuple
    return BoxCords(
        (width_cords[0], length_cords[1]),  # top left corner in (x, y)
        (width_cords[1], length_cords[1]),  # top right corner in (x, y)
        (width_cords[0], length_cords[0]),  # lower left corner in (x, y)
        (width_cords[0], length_cords[1]),  # lower right corner in (x, y)
    )


def overlay_bb(image: np.ndarray, bounding_box: BoxCords) -> Any:
    """
     # Inputs:
    image: a numpy nd array of the image
    bounding_box: the namedtuple bounding box object that contained the defined bounding box corners

    :return: the results of performing the plantcv region of interest rectangle method
    """
    bw_width = bounding_box.ur_corner[0] - bounding_box.ul_corner[0]
    bw_length = bounding_box.lr_corner[1] - bounding_box.ur_corner[1]
    roi1, roi_hierarchy = pcv.roi.rectangle(
        img=image,
        x=bounding_box.ul_corner[0],
        y=bounding_box.ul_corner[1],
        h=bw_length,
        w=bw_width,
    )
    return roi1, roi_hierarchy


def do_segmentation(img: np.ndarray, dark_background: bool = True) -> PlantFeature:
    """
    Perform the image segmentation pipeline to extract the plant from the background.

    :param img: a numpy ndarray defining the image
    :param dark_background: a boolean defining if the background is darker or lighter than the plant
    :return: the plant object extracted using plantcv
    """

    if dark_background:
        object_brightness = "light"
    else:
        object_brightness = "dark"

    # First, convert the image to get hue values...
    hue_extraction = pcv.rgb2gray_hsv(rgb_img=img, channel="v")

    # ... then fill the image based on hue...
    hue_threshold = pcv.threshold.binary(
        gray_img=hue_extraction,
        threshold=100,
        max_value=255,
        object_type="light",
    )

    # ... and perform a slight blur.
    hue_blur = pcv.median_blur(gray_img=hue_threshold, ksize=5)

    # Next, extract a different color family, specifically the green/magenta family...
    gm_image = pcv.rgb2gray_lab(
        rgb_img=img, channel="a"
    )  # a is green/magenta, b is blue/yellow

    # ... and threshold the blue channel image.
    gm_thresh = pcv.threshold.binary(
        gray_img=gm_image, threshold=115, max_value=255, object_type="dark"
    )

    # ... do a quick fill of small objects...
    filled_gm_thresh = pcv.fill(bin_img=gm_thresh, size=100)

    # Next, XOR these two images and get the resulting mask...
    # background_area = pcv.logical_xor(bin_img1=hue_blur, bin_img2=filled_gm_thresh)
    masked = pcv.apply_mask(img=img, mask=filled_gm_thresh, mask_color="white")

    # Next, get the object that is masked from the image...
    id_objects, obj_hierarchy = pcv.find_objects(img=masked, mask=filled_gm_thresh)

    # ... create the boundingbox and overlay on the image...
    bounding_box = create_central_bounding_box(img)
    region, region_her = overlay_bb(masked, bounding_box)

    # ... and keep only the unmasked region in the bounding box...
    # Inputs:
    #    img            = img to display kept objects
    #    roi_contour    = contour of roi, output from any ROI function
    #    roi_hierarchy  = contour of roi, output from any ROI function
    #    object_contour = contours of objects, output from pcv.find_objects function
    #    obj_hierarchy  = hierarchy of objects, output from pcv.find_objects function
    #    roi_type       = 'partial' (default, for partially inside the ROI), 'cutto', or
    #                     'largest' (keep only largest contour)
    region_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(
        img=img,
        roi_contour=region,
        roi_hierarchy=region_her,
        object_contour=id_objects,
        obj_hierarchy=obj_hierarchy,
    )
    obj, mask = pcv.object_composition(
        img=img, contours=region_objects, hierarchy=hierarchy
    )

    image_analysis = pcv.analyze_object(img=img, obj=obj, mask=mask, label="default")

    return PlantFeature(
        plant_obj=None, bg_mask=mask, rgb_image=img, analyzed_image=image_analysis
    )


def do_color_analysis(
    plant_data: PlantFeature, plot: bool = False, plot_title: Optional[str] = None
) -> Any:
    """
    Analyze the color spectrum of an image
    :param plant_data: the namedtuple object that store the extracted image and the mask of the plant in the photo
    :param plot: boolean to define if to plot the results or not
    :param image_name: name to use for the image when plotting

    :return: a dataframe of color frequencies
    """

    if plot_title:
        title_usage = True
    else:
        title_usage = False

    color_analysis_hist = pcv.analyze_color(
        rgb_img=plant_data.rgb_image,
        mask=plant_data.bg_mask,
        colorspaces="all",
        label="default",
    )

    color_data = color_analysis_hist.data
    cols = [col.lower().replace(" ", "_") for col in color_data.columns]

    color_data.columns = cols

    color_data = color_data.pivot(index=cols[0], columns=cols[1], values=cols[2])

    if plot:
        color_data.plot_bokeh(alpha=0.9, figsize=[700, 700], title=f"Color Spectrum for {plot_title}" * title_usage)

    return color_analysis_hist
