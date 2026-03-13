from batchgeneratorsv2.transforms.base.basic_transform import SegOnlyTransform
from typing import Tuple, List, Union
import torch

# this function pulls the segmentation layers apart and puts each one in the list
# this allows us to calculate loss on intermediate outputs
# we expect the segmentation array from the dataloader to follow the order of the cascade, ie
# for a (2, z, h, w), channel 0 is the output from network 0, and channel 1 is the output from network 1
class PullSegApartForCascadeDSTransform(SegOnlyTransform):
    def __init__(self):
        super().__init__()

    def _apply_to_segmentation(self, segmentation: torch.Tensor, **params) -> List[torch.Tensor]:
        results = []
        # loop through channel (first dimension) in segmentation, splitting each tensor into a list
        for chanIdx in range(segmentation.shape[0]):
                results.append(segmentation[chanIdx:chanIdx+1]) # using this slicing preserves the channel dimension
        return results