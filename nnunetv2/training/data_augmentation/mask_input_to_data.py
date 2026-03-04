from typing import Union, List, Tuple

import torch

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform


class MoveInputMaskBackToInput(BasicTransform):
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
        # grab mask inputs from end of segmentation 
        mask_inputs = data_dict['segmentation'][-len(self.source_channel_idx):]
        mask_inputs = mask_inputs.type(data_dict['image'].dtype) # convert to match image data type
        
        # put input masks back into the correct channels in input
        og_index_list = list( set(list(range(data_dict['image'].size[0] + len(self.source_channel_idx)))) - set(self.source_channel_idx) )
        all_inputs = torch.zeros((data_dict['image'].size[0] + len(self.source_channel_idx), *data_dict['image'].size[1:] ))
        all_inputs[self.source_channel_idx] = mask_inputs
        all_inputs[og_index_list] = data_dict['image']
        data_dict['image'] = all_inputs
        
        # delete mask inputs from segmentation
        fin_keep_idx = (data_dict['segmentation'].shape[0] - len(self.source_channel_idx)) + 1
        data_dict['segmentation'] = data_dict['segmentation'][0:fin_keep_idx]
        
        return data_dict