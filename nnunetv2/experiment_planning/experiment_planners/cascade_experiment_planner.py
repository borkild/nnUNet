import shutil
from copy import deepcopy
from typing import List, Union, Tuple
import os

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
        
        # build cascade keyword args
        
        
        # build other config parameters -- these mainly come from other plan files in the cascade
        
        # save out plan as json
        
        
        
        # we use this as a cache to prevent having to instantiate the architecture too often. Saves computation time
        _tmp = {}

        # first get transpose
        transpose_forward, transpose_backward = self.determine_transpose()

        # get fullres spacing and transpose it
        fullres_spacing = self.determine_fullres_target_spacing()
        fullres_spacing_transposed = fullres_spacing[transpose_forward]

        # get transposed new median shape (what we would have after resampling)
        new_shapes = [compute_new_shape(j, i, fullres_spacing) for i, j in
                      zip(self.dataset_fingerprint['spacings'], self.dataset_fingerprint['shapes_after_crop'])]
        new_median_shape = np.median(new_shapes, 0)
        new_median_shape_transposed = new_median_shape[transpose_forward]

        approximate_n_voxels_dataset = float(np.prod(new_median_shape_transposed, dtype=np.float64) *
                                             self.dataset_json['numTraining'])
        # only run 3d if this is a 3d dataset
        if new_median_shape_transposed[0] != 1:
            plan_3d_fullres = self.get_plans_for_configuration(fullres_spacing_transposed,
                                                               new_median_shape_transposed,
                                                               self.generate_data_identifier('3d_fullres'),
                                                               approximate_n_voxels_dataset, _tmp)
            # maybe add 3d_lowres as well
            patch_size_fullres = plan_3d_fullres['patch_size']
            median_num_voxels = np.prod(new_median_shape_transposed, dtype=np.float64)
            num_voxels_in_patch = np.prod(patch_size_fullres, dtype=np.float64)

            plan_3d_lowres = None
            lowres_spacing = deepcopy(plan_3d_fullres['spacing'])

            spacing_increase_factor = 1.03  # used to be 1.01 but that is slow with new GPU memory estimation!
            while num_voxels_in_patch / median_num_voxels < self.lowres_creation_threshold:
                # we incrementally increase the target spacing. We start with the anisotropic axis/axes until it/they
                # is/are similar (factor 2) to the other ax(i/e)s.
                max_spacing = max(lowres_spacing)
                if np.any((max_spacing / lowres_spacing) > 2):
                    lowres_spacing[(max_spacing / lowres_spacing) > 2] *= spacing_increase_factor
                else:
                    lowres_spacing *= spacing_increase_factor
                median_num_voxels = np.prod(plan_3d_fullres['spacing'] / lowres_spacing * new_median_shape_transposed,
                                            dtype=np.float64)
                # print(lowres_spacing)
                plan_3d_lowres = self.get_plans_for_configuration(lowres_spacing,
                                                                  tuple([round(i) for i in plan_3d_fullres['spacing'] /
                                                                         lowres_spacing * new_median_shape_transposed]),
                                                                  self.generate_data_identifier('3d_lowres'),
                                                                  float(np.prod(median_num_voxels) *
                                                                        self.dataset_json['numTraining']), _tmp)
                num_voxels_in_patch = np.prod(plan_3d_lowres['patch_size'], dtype=np.int64)
                print(f'Attempting to find 3d_lowres config. '
                      f'\nCurrent spacing: {lowres_spacing}. '
                      f'\nCurrent patch size: {plan_3d_lowres["patch_size"]}. '
                      f'\nCurrent median shape: {plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed}')
            if np.prod(new_median_shape_transposed, dtype=np.float64) / median_num_voxels < 2:
                print(f'Dropping 3d_lowres config because the image size difference to 3d_fullres is too small. '
                      f'3d_fullres: {new_median_shape_transposed}, '
                      f'3d_lowres: {[round(i) for i in plan_3d_fullres["spacing"] / lowres_spacing * new_median_shape_transposed]}')
                plan_3d_lowres = None
            if plan_3d_lowres is not None:
                plan_3d_lowres['batch_dice'] = False
                plan_3d_fullres['batch_dice'] = True
            else:
                plan_3d_fullres['batch_dice'] = False
        else:
            plan_3d_fullres = None
            plan_3d_lowres = None

        # 2D configuration
        plan_2d = self.get_plans_for_configuration(fullres_spacing_transposed[1:],
                                                   new_median_shape_transposed[1:],
                                                   self.generate_data_identifier('2d'), approximate_n_voxels_dataset,
                                                   _tmp)
        plan_2d['batch_dice'] = True

        print('2D U-Net configuration:')
        print(plan_2d)
        print()

        # median spacing and shape, just for reference when printing the plans
        median_spacing = np.median(self.dataset_fingerprint['spacings'], 0)[transpose_forward]
        median_shape = np.median(self.dataset_fingerprint['shapes_after_crop'], 0)[transpose_forward]

        # instead of writing all that into the plans we just copy the original file. More files, but less crowded
        # per file.
        shutil.copy(join(self.raw_dataset_folder, 'dataset.json'),
                    join(nnUNet_preprocessed, self.dataset_name, 'dataset.json'))

        # json is ###. I hate it... "Object of type int64 is not JSON serializable"
        plans = {
            'dataset_name': self.dataset_name,
            'plans_name': self.plans_identifier,
            'original_median_spacing_after_transp': [float(i) for i in median_spacing],
            'original_median_shape_after_transp': [int(round(i)) for i in median_shape],
            'image_reader_writer': self.determine_reader_writer().__name__,
            'transpose_forward': [int(i) for i in transpose_forward],
            'transpose_backward': [int(i) for i in transpose_backward],
            'configurations': {'2d': plan_2d},
            'experiment_planner_used': self.__class__.__name__,
            'label_manager': 'LabelManager',
            'foreground_intensity_properties_per_channel': self.dataset_fingerprint[
                'foreground_intensity_properties_per_channel']
        }

        if plan_3d_lowres is not None:
            plans['configurations']['3d_lowres'] = plan_3d_lowres
            if plan_3d_fullres is not None:
                plans['configurations']['3d_lowres']['next_stage'] = '3d_cascade_fullres'
            print('3D lowres U-Net configuration:')
            print(plan_3d_lowres)
            print()
        if plan_3d_fullres is not None:
            plans['configurations']['3d_fullres'] = plan_3d_fullres
            print('3D fullres U-Net configuration:')
            print(plan_3d_fullres)
            print()
            if plan_3d_lowres is not None:
                plans['configurations']['3d_cascade_fullres'] = {
                    'inherits_from': '3d_fullres',
                    'previous_stage': '3d_lowres'
                }

        self.plans = plans
        self.save_plans(plans)
        return plans

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
