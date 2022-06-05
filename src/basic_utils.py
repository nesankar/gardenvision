import cv2
import glob
import sys
from typing import Union, List, Any, Optional, Dict
from collections import namedtuple
from pathlib import Path
import numpy as np
import pandas_bokeh as pbk
from plantcv import plantcv as pcv
import matplotlib


"""Simple processing utilities"""

file_dir = Path(__file__)
root = file_dir.parent.parent
sys.path.append(str(root))

# PlantImage = namedtuple("PlantImage", ["image", "name"])
BoxCords = namedtuple("BoxCords", ["ul_corner", "ur_corner", "ll_corner", "lr_corner"])
PlantFeature = namedtuple(
    "PlantFeature", ["plant_obj", "bg_mask", "rgb_image", "analyzed_image", "area"]
)
ReferenceFeature = namedtuple(
    "ReferenceFeature", ["reference_obj", "bg_mask", "max_pxl_length", "max_in_length"]
)
ColorThreshold = namedtuple(
    "ColorTheshold", ["by_threshold", "gm_threshold", "by_obj_type", "gm_obj_type"]
)

REF_THRESHOLDS: Dict[str, ColorThreshold] = {
    "blue": ColorThreshold(115, 105, "dark", "dark"),
    "yellow": ColorThreshold(190, 125, "light", "dark"),
}


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


def find_reference_obj(
    plant_img: np.ndarray,
    image_name: str,
    obj_color: str,
    max_length_dim_inches: float = 2.0,
) -> ReferenceFeature:
    """
    Perform the segmentation to pull out the reference object in the image. Assumes a blue reference obj.

    :param plant_img: a numpy ndarray defining the image
    :param image_name: a string uniquely identifying the image
    :param max_length_dim_inches: a float equal to the longest length/width dimension of the reference obj
    :return: the reference object extracted.
    """

    # Get some constants
    ref_obj_label = f"{image_name}_ref_obj"
    object_thresholds = REF_THRESHOLDS[obj_color]

    # First take the blue/yellow image...
    by_image = pcv.rgb2gray_lab(rgb_img=plant_img, channel="b")

    # ... then the green magenta image
    gm_image = pcv.rgb2gray_lab(
        rgb_img=plant_img, channel="a"
    )  # a is green/magenta, b is blue/yellow

    # ... and threshold the green/magenta and blue/yellow channel images.
    by_thresh = pcv.threshold.binary(
        gray_img=by_image,
        threshold=object_thresholds.by_threshold,
        max_value=225,
        object_type=object_thresholds.by_obj_type,
    )
    gm_thresh = pcv.threshold.binary(
        gray_img=gm_image,
        threshold=object_thresholds.gm_threshold,
        max_value=225,
        object_type=object_thresholds.gm_obj_type,
    )

    # Now get the object masked by the image...
    joined_mask = pcv.logical_and(bin_img1=by_thresh, bin_img2=gm_thresh)

    # ... and do a small bit of filling before masking.
    joined_mask = pcv.fill(bin_img=joined_mask, size=1000)

    masked = pcv.apply_mask(img=plant_img, mask=joined_mask, mask_color="white")
    reference_objects, obj_hierarchy = pcv.find_objects(img=masked, mask=joined_mask)

    # Compose the object
    obj, mask = pcv.object_composition(
        img=plant_img, contours=reference_objects, hierarchy=obj_hierarchy
    )

    pcv.analyze_object(
        img=plant_img, obj=obj, mask=mask, label=ref_obj_label
    )  # pcv stores data for each analysis internally via a dict with the "label" as the key

    max_pixel_length = max(
        [
            pcv.outputs.observations[ref_obj_label][dimension]["value"]
            for dimension in ["width", "height"]
        ]
    )

    return ReferenceFeature(obj, mask, max_pixel_length, max_length_dim_inches)


def do_plant_segmentation(img: np.ndarray, image_name: str) -> PlantFeature:
    """
    Perform the image segmentation pipeline to extract the plant from the background.

    :param img: a numpy ndarray defining the image
    :return: the plant object extracted using plantcv
    """

    plant_label = f"{image_name}_plant_obj"

    # First, extract different color families, specifically the green/magenta family...
    gm_image = pcv.rgb2gray_lab(
        rgb_img=img, channel="a"
    )  # a is green/magenta, b is blue/yellow

    # ... also take the blue/yellow image...
    by_image = pcv.rgb2gray_lab(rgb_img=img, channel="b")

    # ... and threshold the green/magenta and blue/yellow channel images.
    gm_thresh = pcv.threshold.binary(
        gray_img=gm_image, threshold=115, max_value=255, object_type="dark"
    )
    by_thresh = pcv.threshold.binary(
        gray_img=by_image, threshold=150, max_value=255, object_type="light"
    )

    # ... do a quick fill of small objects...
    filled_gm_thresh = pcv.fill(bin_img=gm_thresh, size=100)
    filled_by_thresh = pcv.fill(bin_img=by_thresh, size=5000)

    # Next, XOR these two images and get the resulting mask...
    joined_mask = pcv.logical_or(bin_img1=filled_by_thresh, bin_img2=filled_gm_thresh)
    masked = pcv.apply_mask(img=img, mask=joined_mask, mask_color="white")

    # Next, get the object that is masked from the image...
    id_objects, obj_hierarchy = pcv.find_objects(img=masked, mask=joined_mask)

    # ... create the boundingbox and overlay on the image...
    bounding_box = create_central_bounding_box(img)
    region, region_her = overlay_bb(masked, bounding_box)

    # ... and keep only the unmasked region in the bounding box...
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

    image_analysis = pcv.analyze_object(img=img, obj=obj, mask=mask, label=plant_label)

    # pcv.plot_image(image_analysis)

    return PlantFeature(
        plant_obj=None,
        bg_mask=mask,
        rgb_image=img,
        analyzed_image=image_analysis,
        area=obj_area,
    )


def do_color_analysis(
    plant_data: PlantFeature, plot: bool = False, plot_title: Optional[str] = None
) -> Any:
    """
    Analyze the color spectrum of an image
    :param plant_data: the namedtuple object that store the extracted image and the mask of the plant in the photo
    :param plot: boolean to define if to plot the results or not
    :param plot_title: name to use for the image when plotting

    :return: a dataframe of color frequencies
    """

    if plot_title:
        title_usage = True
    else:
        title_usage = False

    # Set this to NOT print out anything which always errors heavy
    matplotlib.use("Agg")

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
        color_data[["blue", "blue-yellow", "green", "green-magenta", "red"]].plot_bokeh(
            alpha=0.9,
            figsize=[700, 700],
            ylim=[0, 13],
            title=f"Color Spectrum for {plot_title}" * title_usage,
        )

    # And reset on the way out...
    matplotlib.use("macosx")
    return color_analysis_hist
