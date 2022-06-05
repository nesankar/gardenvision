from typing import Optional, Dict, Any
import pandas as pd
from plantcv import plantcv as pcv
import numpy as np
import logging
from src.basic_utils import (
    do_plant_segmentation,
    find_reference_obj,
    PlantFeature,
    do_color_analysis,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PlantImage:
    """Contain the data for a specific image of a plant."""

    def __init__(
        self,
        image: np.ndarray,
        name: str,
        ref_obj_color: str,
        reference_obj_max_dim_in: Optional[float] = None,
    ):

        self.image = image
        self.name = name
        self.ref_obj_color = ref_obj_color
        self.reference_obj_max_dim_in = reference_obj_max_dim_in

    @property
    def reference_length(self) -> float:
        if hasattr(self, "_reference_length"):
            if self._reference_length == 0.0:
                raise Warning(
                    "No object length was provided. Using the ._refernce_length attribute is not possible."
                )
            return self._reference_length
        else:
            self._reference_length = self.get_reference_length()
            return self._reference_length

    def get_reference_length(self) -> float:
        """
        For the image with the reference object, get the reference length of pixels/inch
        :return: the pixels/inch ratio for the image
        """
        if not self.reference_obj_max_dim_in:
            logger.info(
                f"\n No reference dimension provided for image: {self.name}. Can not get a reference length."
            )
            return 0.0

        logger.info(f"Calculating reference object length fpr {self.name}")
        reference_feature = find_reference_obj(
            self.image, self.name, self.ref_obj_color, self.reference_obj_max_dim_in
        )

        return reference_feature.max_pxl_length / reference_feature.max_in_length

    @property
    def plant_object(self) -> PlantFeature:
        if hasattr(self, "_plant_object"):
            return self._plant_object
        else:
            self.extract_plant()
            return self._plant_object

    @property
    def plant_size_data(self) -> Dict[str, Any]:
        if hasattr(self, "_plant_size_data"):
            return self._plant_size_data
        else:
            self.extract_plant()
            return self._plant_size_data

    def extract_plant(self) -> None:
        """
        Extract the plant from the image, providing a PlantFeature object. Simply a wrapper around the segmentation
        function to set some attributes.
        :return: PlantFeature object for the plant in the image
        """
        # Extract the plant from the image...
        logger.info(f"Performing plant segmentation for {self.name}")
        self._plant_object = do_plant_segmentation(self.image, self.name)

        # and also get the plant image size data out of the plant_cv outputs
        self._plant_size_data = pcv.outputs.observations[f"{self.name}_plant_obj"]

    @property
    def plant_width_in(self) -> float:
        if hasattr(self, "_plant_width_in"):
            return self._plant_width_in
        else:
            self.get_plant_sizes()
            return self._plant_width_in

    @property
    def plant_length_in(self) -> float:
        if hasattr(self, "_plant_width_in"):
            return self._plant_length_in
        else:
            self.get_plant_sizes()
            return self._plant_length_in

    @property
    def plant_area_in2(self) -> float:
        if hasattr(self, "_plant_width_in"):
            return self._plant_area_in2
        else:
            self.get_plant_sizes()
            return self._plant_area_in2

    def get_plant_sizes(self) -> None:
        """Get the size of the current plant in inches, and the pct of that area "filled" with plant.
        Note, this is only a 2-D "from above" image.
        :return:
        """

        plant_width_pxl = self._plant_size_data["width"]["value"]
        plant_length_pxl = self._plant_size_data["height"][
            "value"
        ]  # height seems to be more vertical? using length

        plant_width_inches = plant_width_pxl / self.reference_length
        plant_length_inches = plant_length_pxl / self.reference_length

        plant_total_area = plant_width_inches * plant_length_inches

        plant_pixel_area = self.plant_object.area  # in pixels
        fraction_growth = plant_pixel_area / (plant_length_pxl * plant_length_pxl)

        self._plant_width_in = plant_width_inches
        self._plant_length_in = plant_length_inches
        self._plant_growth_fraction = fraction_growth
        self._plant_area_in2 = plant_total_area * fraction_growth

    @property
    def color_spectrum_df(self) -> pd.DataFrame:
        if hasattr(self, "_color_spectrum_df"):
            return self._color_spectrum_df
        else:
            self.get_plant_color_spectrum(plot=False)  # don't want to plot in this case
            return self._color_spectrum_df

    def get_plant_color_spectrum(self, plot: bool = True) -> None:
        """
        Analyze the color spectrum of an image using the plant image and the mask of the plant in the photo. Simply
        wrap the basic_utils.do_color_analysis function, and set the attribute for the class.
        :param plot: boolean to designate if the color spectrum should be plotted
        :return: None
        """
        logger.info(f"Getting color spectrum analysis for {self.name}")
        color_spectrum_df = do_color_analysis(
            self.plant_object, plot=plot, plot_title=self.name
        )
        self._color_spectrum_df = color_spectrum_df

    def plot_analyzed_plant(self) -> None:
        """Just plot out the analyzed plant image"""
        pcv.plot_image(self.plant_object.analyzed_image)

    def plot_plant(self) -> None:
        """Just plot out the plant image"""
        pcv.plot_image(self.plant_object.plant_obj)
