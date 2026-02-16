import shutil
from copy import deepcopy
from typing import List, Union, Tuple
import os
import warnings

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.preprocessing.normalization.map_channel_name_to_normalization import get_normalization_scheme
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner

# make our class a child of the default experiment planner so that we don't need to have a bunch of unnecessary functions here
class CascadeExperimentPlanner(ExperimentPlanner):
    def __init__(self, dataset_name_or_id: Union[str, int],
                 gpu_memory_target_in_gb: float = 8,
                 preprocessor_name: str = 'DefaultPreprocessor', plans_name: str = 'nnUNetCascadePlans',
                 overwrite_target_spacing: Union[List[float], Tuple[float, ...]] = None,
                 suppress_transpose: bool = False,
                 cascade_networks: list = [],
                 cascade_configs: list = [],
                 checkpoint_weight_tag: str = "final" # will expose this option upstream later
                 ):
        """
        overwrite_target_spacing only affects 3d_fullres! (but by extension 3d_lowres which starts with fullres may
        also be affected
        """

        self.dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.suppress_transpose = suppress_transpose
        self.raw_dataset_folder = join(nnUNet_raw, self.dataset_name)
        preprocessed_folder = join(nnUNet_preprocessed, self.dataset_name)
        self.dataset_json = load_json(join(self.raw_dataset_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.raw_dataset_folder, self.dataset_json)
        
        # load dataset names into list
        self.network_datasets = []
        for netIdx in range(len(cascade_networks)):
            self.network_datasets.append( maybe_convert_to_dataset_name(cascade_networks[netIdx]) )
            
        self.configs = cascade_configs
        self.checkpoint_weight_tag = checkpoint_weight_tag

        # load dataset fingerprint
        if not isfile(join(preprocessed_folder, 'dataset_fingerprint.json')):
            raise RuntimeError('Fingerprint missing for this dataset. Please run nnUNet_extract_dataset_fingerprint')

        self.dataset_fingerprint = load_json(join(preprocessed_folder, 'dataset_fingerprint.json'))

        self.anisotropy_threshold = ANISO_THRESHOLD

        self.UNet_base_num_features = 32
        self.UNet_class = PlainConvUNet
        # the following two numbers are really arbitrary and were set to reproduce nnU-Net v1's configurations as
        # much as possible
        self.UNet_reference_val_3d = 560000000  # 455600128  550000000
        self.UNet_reference_val_2d = 85000000  # 83252480
        self.UNet_reference_com_nfeatures = 32
        self.UNet_reference_val_corresp_GB = 8
        self.UNet_reference_val_corresp_bs_2d = 12
        self.UNet_reference_val_corresp_bs_3d = 2
        self.UNet_featuremap_min_edge_length = 4
        self.UNet_blocks_per_stage_encoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_blocks_per_stage_decoder = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        self.UNet_min_batch_size = 2
        self.UNet_max_features_2d = 512
        self.UNet_max_features_3d = 320
        self.max_dataset_covered = 0.05 # we limit the batch size so that no more than 5% of the dataset can be seen
        # in a single forward/backward pass

        self.UNet_vram_target_GB = gpu_memory_target_in_gb

        self.lowres_creation_threshold = 0.25  # if the patch size of fullres is less than 25% of the voxels in the
        # median shape then we need a lowres config as well

        self.preprocessor_name = preprocessor_name
        self.plans_identifier = plans_name
        self.overwrite_target_spacing = overwrite_target_spacing
        assert overwrite_target_spacing is None or len(overwrite_target_spacing), 'if overwrite_target_spacing is ' \
                                                                                  'used then three floats must be ' \
                                                                                  'given (as list or tuple)'
        assert overwrite_target_spacing is None or all([isinstance(i, float) for i in overwrite_target_spacing]), \
            'if overwrite_target_spacing is used then three floats must be given (as list or tuple)'

        self.plans = None

        if isfile(join(self.raw_dataset_folder, 'splits_final.json')):
            _maybe_copy_splits_file(join(self.raw_dataset_folder, 'splits_final.json'),
                                    join(preprocessed_folder, 'splits_final.json'))


    def build_cascade_arch_plan(self):
        # this function returns everything that belongs in the "architecture" key in the cascaded plan dict
        networks = {}
        # iterate through networks in cascade
        for netIdx in range(len(self.network_datasets)):
            networks["network_"+str(netIdx)] = {}
            # check for plan file in given folder -- for now we assume
            if os.path.isfile( os.path.join(nnUNet_preprocessed, self.network_datasets[netIdx], "nnUNetPlans.json") ):
                networks["network_"+str(netIdx)]["plan_file_path"] = os.path.join(nnUNet_preprocessed, self.network_datasets[netIdx], "nnUNetPlans.json")
            else:
                print("couldn't find plan for network " + str(netIdx) + " in cascade")
                raise ImportError()
            # check for matching config in this network's plans file
            cur_plan_dict = load_json( os.path.join(nnUNet_preprocessed, self.network_datasets[netIdx], "nnUNetPlans.json") )
            cur_plan_configs = cur_plan_dict["configurations"]
            if self.configs[netIdx] in cur_plan_configs:
                networks["network_"+str(netIdx)]["network_config"] = self.configs[netIdx]
            else:
                print("Chosen configuration is not in the plan file for network " + str(netIdx))
                raise ImportError()
            # find folder of trained weights (from individual network training) for each fold -- again, we assume nnUnet plans and trainer used for now
            if os.path.isdir( os.path.join(nnUNet_results, self.network_datasets[netIdx], "nnUNetTrainer__nnUNetPlans__" + self.configs[netIdx]) ):
                networks["network_"+str(netIdx)]["weight_save_path"] = os.path.join(nnUNet_results, self.network_datasets[netIdx], 
                                                                                    "nnUNetTrainer__nnUNetPlans__"+self.configs[netIdx])
            else:
                print("Can't find results folder with saved networks for network " + str(netIdx) + " config " + self.configs[netIdx])
            # add desired weight config type
            networks["network_"+str(netIdx)]["weight_save_type"] = self.checkpoint_weight_tag
            # now load dataset json file, and grab number of input and output channels
            dataset = load_json( os.path.join(self.raw_dataset_folder, "dataset.json") )
            networks["network_"+str(netIdx)]["num_input_channels"] = len(dataset["channel_names"])
            networks["network_"+str(netIdx)]["num_output_classes"] = len(dataset["labels"])
        # return dict
        return networks
            
    # this function checks to make sure the train and validation splits are the same for all networks
    # we don't want data leak by having scans in the train set in the validation set for another network
    def check_train_val_splits(self):
        later_networks = []
        # iterate through networks, loading split.json file
        for netIdx in range(len(self.network_datasets)):
            if netIdx == 0:
                first_network_split = load_json( os.path.join(nnUNet_preprocessed, self.network_datasets[netIdx], "splits_final.json") )
            else:
                later_networks.append( load_json( os.path.join(nnUNet_preprocessed, self.network_datasets[netIdx], "splits_final.json") ) )

        # iterate through folds
        for foldIdx in range(len(first_network_split)):
            first_net_set = set(first_network_split[foldIdx]["val"]) # only need to grab and check validation set
            later_net_sets = []
            for latIdx in range(len(later_networks)):
                later_net_sets.append( set(later_networks[latIdx]["val"]) )
            # check each validation set to match -- sets don't care about order, we make use of them here
            sets_equal = all( x == first_net_set for x in later_net_sets )
            
            # if not all the sets match, then we throw an error, as this will mess up our training and validation metrics
            if not all:
                print("Different scans in validation set of fold " + str(foldIdx))
                print("Please retrain every network in the cascade with the same train/validation splits.")
                raise ValueError()
        
        # check if preprocessed folder exists, if not, make it
        if not os.path.isdir( os.path.join(nnUNet_preprocessed, self.dataset_name) ):
            os.mkdir(os.path.join(nnUNet_preprocessed, self.dataset_name))
        
        # if we made it here, then the train/validation splits match, so we copy the splits file from the first network
        # to the preprocessed folder
        shutil.copy2(os.path.join(nnUNet_preprocessed, self.network_datasets[netIdx], "splits_final.json"), os.path.join(nnUNet_preprocessed, self.dataset_name) )
        
        print("train/validation splits were valid, splits were copied to cascaded dataset")
    
    
    
    # this function checks the train and test splits to verify they are the same in each dataset in the cascade
    # This prevents data leaks
    def check_train_test_sets(self):
        # get lists of files in dataset folders
        first_train_dataset = os.listdir( os.path.join(nnUNet_raw, self.network_datasets[0], "imagesTr") )
        first_train_dataset = set([file for file in first_train_dataset if "_0000" in file]) # filter inputs to only grab channel 0 and convert to set
        first_test_dataset = os.listdir( os.path.join(nnUNet_raw, self.network_datasets[0], "imagesTs") )
        first_test_dataset = set([file for file in first_test_dataset if "_0000" in file]) # filter inputs to only grab channel 0
        all_other_train = []
        all_other_test = []
        # iterate through all other networks in dataset, grabbing train and test sets
        for netIdx in range(1, len(self.network_datasets)):
            tempTrain = os.listdir( os.path.join(nnUNet_raw, self.network_datasets[netIdx], "imagesTr") )
            all_other_train.append( set([file for file in tempTrain if "_0000" in file]) )
            tempTest = os.listdir( os.path.join(nnUNet_raw, self.network_datasets[0], "imagesTs") )
            all_other_test.append( set([file for file in tempTest if "_0000" in file]) )
            
        # compare train and test sets for each network, making sure they match
        train_sets_equal = all(x == first_train_dataset for x in all_other_train)
        test_sets_equal = all(x == first_test_dataset for x in all_other_test)
        
        if not train_sets_equal:
            print("Train sets are not the same! Please retrain networks with the same train/test splits before starting cascade fine tuning.")
            raise ImportError()
        elif not test_sets_equal:
            print("Test sets are not the same! Please retrain networks with the same train/test splits before starting cascade fine tuning.")
            raise ImportError()
        else:
            print("Train and test sets are the same for all networks! Moving to next planner step.")
        
    
    def gen_cascade_kwargs(self, arch_dict):
        # here we generate the architecture keyword args that are needed for the 
        # cascaded trainer to build the architecture
        kwargs = {}
        no_diff = False
        input_diff = True
        for netIdx in range(len(arch_dict)-2):
            # check to see if the input will be passed to each network
            if netIdx == 0:
                input_channels = arch_dict["network_"+str(netIdx)]["num_input_channels"]
            cur_output_classes = arch_dict["network_"+str(netIdx)]["num_output_classes"]
            nxt_input_channels = arch_dict["network_"+str(netIdx+1)]["num_input_channels"]
            chan_diff = nxt_input_channels - cur_output_classes
            # check for difference in number of channels from one network to another
            if chan_diff == 0:
                no_diff = True
            elif chan_diff == input_channels:
                input_diff = True
            else:
                print(f"Dimension mismatch! Networks {netIdx} and {netIdx+1} do not have "
                      "compatible input and output sizes")
                raise ValueError()
        # now check to make sure we consistently hand the input to the network
        if no_diff and input_diff:
            print("Error! Some network appear to get the original input, while others do not. "
                  "Mixing networks that do and don't recieve the input in a cascade is currently "
                  "not supported.")
        elif no_diff:
            kwargs["input_to_all_networks"] = False
        elif input_diff:
            kwargs["input_to_all_networks"] = True

        # now we add the other parameters
        # for now, I am setting these mannually, will expose to command line evenetually
        kwargs["intermediate_transforms"] = None
        kwargs["intermediate_outputs"] = False
        kwargs["split_intermediate_outputs"] = True
        
                
    def grab_other_config_details(self, arch_dict):
        # grab plan file for first network
        other_config_details = {}
        net0_plan = load_json(arch_dict["network_0"]["plan_file_path"])
        net0_config = net0_plan["configurations"][arch_dict["network_0"]["network_config"]]
        # list of keys to grab from network 0 config
        keys_to_grab = ["data_identifier",
                        "preprocessor_name",
                        "batch_size",
                        "patch_size",
                        "median_image_size_in_voxels",
                        "spacing",
                        "normalization_schemes",
                        "use_mask_for_norm",
                        "resampling_fn_data",
                        "resampling_fn_seg",
                        "resampling_fn_data_kwargs",
                        "resampling_fn_seg_kwargs",
                        "resampling_fn_probabilities",
                        "resampling_fn_probabilities_kwargs"
                        ]
        # build out list of other config details
        for key in keys_to_grab:
            other_config_details[key] = net0_config[key]

        # check patch sizes for all networks -- make sure they are the same dimension
        # and ~ideally~ the same patch size
        for netIdx in range(1, len(arch_dict)-1):
            cur_net_plan = load_json(arch_dict["network_"+str(netIdx)]["plan_file_path"])
            cur_patch_size = cur_net_plan["configuration"][arch_dict["network_"+str(netIdx)]["network_config"]]["patch_size"]
            # check that dimensions match
            if len(other_config_details["patch_size"]) != cur_patch_size:
                print("Input dimensions for networks are not the same! All networks in cascade must be 2D or 3D.")
                raise ValueError()

            # check for overall match -- NOTE: this isn't strictly enforced, 
            # we only output a warning here
            if other_config_details["patch_size"] != cur_patch_size:
                warnings.warn("The patch sizes were not the same during network training. This may effect performance.")

        return other_config_details
   

    
    def plan_experiment(self):
        """
        MOVE EVERYTHING INTO THE PLANS. MAXIMUM FLEXIBILITY

        Ideally I would like to move transpose_forward/backward into the configurations so that this can also be done
        differently for each configuration but this would cause problems with identifying the correct axes for 2d. There
        surely is a way around that but eh. I'm feeling lazy and featuritis must also not be pushed to the extremes.

        So for now if you want a different transpose_forward/backward you need to create a new planner. Also not too
        hard.
        """
        # our cascaded plan won't need to calculate anything new, it just needs to pull info from other plans to form the cascade plan file
        cascaded_plan_dict = {} # dict that we'll eventually write as a .json for the plan
        cascaded_plan_dict["dataset_name"] = self.dataset_name
        cascaded_plan_dict["plans_name"] = self.plans_identifier
        cascaded_plan_dict["image_reader_writer"] = self.determine_reader_writer().__name__
        
        # check to make sure train and test sets match for each network
        self.check_train_test_sets()
        
        # check to make sure the previous train/val splits match for each network
        self.check_train_val_splits()
        
        cascaded_plan_dict["cascade_config"] = {}
        # build architecture part of plan
        arch_config = self.build_cascade_arch_plan()
        cascaded_plan_dict["cascade_config"]["architecture"] = arch_config
        cascaded_plan_dict["cascade_config"]["number_of_networks"] = len(cascaded_plan_dict["cascade_config"]["architecture"]) - 1
        
        # build cascade keyword args
        cascade_kwargs = self.gen_cascade_kwargs(arch_config)
        cascaded_plan_dict["cascade_config"]["arch_kwargs"] = cascade_kwargs
        
        # build other config parameters -- these mainly come from other plan files in the cascade
        other_config_details = self.grab_other_config_details(arch_config)
        cascaded_plan_dict["cascade_config"] = {**cascaded_plan_dict["cascade_config"], **other_config_details}
        

        # save out plan as json
        self.plans = cascaded_plan_dict
        self.save_plans(cascaded_plan_dict)
        return cascaded_plan_dict
        

    def save_plans(self, plans):
        recursive_fix_for_json_export(plans)

        plans_file = join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json')

        # we don't want to overwrite potentially existing custom configurations every time this is executed. So let's
        # read the plans file if it already exists and keep any non-default configurations
        if isfile(plans_file):
            old_plans = load_json(plans_file)
            old_configurations = old_plans['configurations']
            for c in plans['configurations'].keys():
                if c in old_configurations.keys():
                    del (old_configurations[c])
            plans['configurations'].update(old_configurations)

        maybe_mkdir_p(join(nnUNet_preprocessed, self.dataset_name))
        save_json(plans, plans_file, sort_keys=False)
        print(f"Plans were saved to {join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + '.json')}")

    def generate_data_identifier(self, configuration_name: str) -> str:
        """
        configurations are unique within each plans file but different plans file can have configurations with the
        same name. In order to distinguish the associated data we need a data identifier that reflects not just the
        config but also the plans it originates from
        """
        return self.plans_identifier + '_' + configuration_name

    def load_plans(self, fname: str):
        self.plans = load_json(fname)


def _maybe_copy_splits_file(splits_file: str, target_fname: str):
    if not isfile(target_fname):
        shutil.copy(splits_file, target_fname)
    else:
        # split already exists, do not copy, but check that the splits match.
        # This code allows target_fname to contain more splits than splits_file. This is OK.
        splits_source = load_json(splits_file)
        splits_target = load_json(target_fname)
        # all folds in the source file must match the target file
        for i in range(len(splits_source)):
            train_source = set(splits_source[i]['train'])
            train_target = set(splits_target[i]['train'])
            assert train_target == train_source
            val_source = set(splits_source[i]['val'])
            val_target = set(splits_target[i]['val'])
            assert val_source == val_target


if __name__ == '__main__':
    CascadeExperimentPlanner(12, cascade_networks=[19, 29], cascade_configs=["3d_fullres", "3d_fullres"]).check_train_val_splits()
