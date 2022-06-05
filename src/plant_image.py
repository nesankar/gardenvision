from typing import Optional, Dict, Any
import plantcv as pcv
import numpy as np
from basic_utils import do_plant_segmentation, find_reference_obj, PlantFeature


class PlantImage:
    """Contain the data for a specific image of a plant."""

    def __init__(
        self,
        image: np.ndarray,
        name: str,
        reference_obj_max_dim_in: Optional[int] = None,
    ):

        self.image = image
        self.name = name
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
            print(
                f"\n No reference dimension provided for image: {self.name}. Can not get a reference length."
            )
            return 0.0

        reference_feature = find_reference_obj(
            self.image, self.name, self.reference_obj_max_dim_in
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
        self._plant_object = do_plant_segmentation(self.image, self.name)

        # and also get the plant image size data out of the plant_cv outputs
        self._plant_size_data = pcv.outputs.observations[f"{self.name}_plant_obj"]
