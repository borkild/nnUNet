from typing import Union, List, Tuple

import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class MoveInputMaskToSegParam(BasicTransform):
    def __init__(self, source_channel_idx: list[int]):
        """
        Used to handle masks given as inputs in nnUnet -- essentially, we will move the input masks to the segmentation key in data_dict,
        then back to the data with mask_input_to_data
        
        This just makes it easy to handle masked inputs without need to change a ton of the codebase, and avoids applying wrong
        data augmentations to input masks.
        
        Args:
            source_channel_idx: indices of mask channels
            all_labels:
            remove_channel_from_source:
        """
        super().__init__()
        self.source_channel_idx = source_channel_idx

    def apply(self, data_dict, **params):
        # get mask inputs
        mask_inputs = data_dict['image'][self.source_channel_idx]
        # convert to same type as segementation
        mask_inputs = mask_inputs.type(data_dict['segmentation'].dtype)
        # stack mask inputs onto segmentation
        data_dict['segmentation'] = torch.cat((data_dict['segmentation'], mask_inputs))
        # get rid of mask input channels in data_dict
        keep_channels = [ i for i in range(data_dict['image'].shape[0]) if not i in self.source_channel_idx]
        data_dict['image'] = data_dict['image'][keep_channels]
        return data_dict