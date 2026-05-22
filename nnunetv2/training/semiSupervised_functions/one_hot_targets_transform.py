import torch
import numpy as np

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform

class oneHotFloatTargets(BasicTransform):
    def __init__(self):
        self.hi = 1
        
    def apply(self, data_dict, **params):
        print(f"seg dimension in one hot conversion: {data_dict["segmentation"].shape}")
        bg_value = torch.ones(data_dict['segmentation'].shape) - torch.sum(data_dict['segmentation'], dim=0, keepdim=True)
        data_dict['segmentation'] = torch.cat( (bg_value, data_dict['segmentation']), dim=0 )
        
        return data_dict
    
'''
if __name__ == "__main__":
    tst1 = torch.zeros((1,1,3,3))
    tst2 = torch.ones((1,1,3,3))
    tst3 = torch.rand((1,1,3,3))
    
    tmp_dict = {"segmentation": tst3}
    
    tmpclass = oneHotFloatTargets
    out = tmpclass.apply(tmpclass, tmp_dict)
    print(out['segmentation'])
    print(out["segmentation"].shape)
    print(torch.sum(out['segmentation'], dim=1))
'''