import multiprocessing
import os
from time import sleep
from typing import List, Type, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json, join, save_json, isfile, maybe_mkdir_p
from tqdm import tqdm

import shutil

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

# make our class a child of the basic class -- this way we don't need to redefine things here
from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor

class cascadeDatasetFingerprintExtractor(DatasetFingerprintExtractor):
    def __init__(self, dataset_name_or_id: Union[str, int], num_processes: int = 8, verbose: bool = False, cascade_network_list: list = []):
        """
        extracts the dataset fingerprint used for experiment planning. The dataset fingerprint will be saved as a
        json file in the input_folder

        Philosophy here is to do only what we really need. Don't store stuff that we can easily read from somewhere
        else. Don't compute stuff we don't need (except for intensity_statistics_per_channel)
        """
        # check to make sure we actually need the cascaded extractor
        if len(cascade_network_list) == 0:
            print("No networks given for cascade, check your inputs or use the basic fingerprint extractor class (DatasetFingerprintExtractor).")
        
        # we don't expect the dataset to exist yet, so we assign it a name here
        if isinstance(dataset_name_or_id, str):
            dataset_name = dataset_name_or_id
        else:
            dataset_name = "Dataset" + str(dataset_name_or_id).zfill(3) + "_cascadeFineTuning"
        
        self.verbose = verbose

        self.dataset_name = dataset_name
        self.input_folder = join(nnUNet_raw, dataset_name)
        self.num_processes = num_processes
        
        # 
        self.cascade_networks = cascade_network_list
        self.cascade_dataset_names = []
        for netIdx in range(len(cascade_network_list)):
            self.cascade_dataset_names.append( maybe_convert_to_dataset_name(self.cascade_networks[netIdx]) )

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory). It is
        # also not critically important to get all pixels as long as there are enough. Let's use 10e7 voxels in total
        # (for the entire dataset)
        self.num_foreground_voxels_for_intensitystats = 10e7

    
    @property
    def gen_cascade_dataset(self):
        # make directory
        os.mkdir( os.path.join(nnUNet_raw, self.dataset_name) )
        print("creating directory: " + os.path.join(nnUNet_raw, self.dataset_name))
        # copy train and test inputs from first network folder
        shutil.copytree( os.path.join(nnUNet_raw, self.cascade_dataset_names[0], "imagesTr"), os.path.join(nnUNet_raw, self.dataset_name, "imagesTr") )
        shutil.copytree( os.path.join(nnUNet_raw, self.cascade_dataset_names[0], "imagesTs"), os.path.join(nnUNet_raw, self.dataset_name, "imagesTs") )
        # copy train and test labels from last network
        shutil.copytree( os.path.join(nnUNet_raw, self.cascade_dataset_names[-1], "labelsTr"), os.path.join(nnUNet_raw, self.dataset_name, "labelsTr") )
        shutil.copytree( os.path.join(nnUNet_raw, self.cascade_dataset_names[-1], "labelsTs"), os.path.join(nnUNet_raw, self.dataset_name, "labelsTs") )
        print("Copied inputs and labels into new dataset for cascaded fine-tuning!")
        
        # check that number of train and test scans match -- we will do a more in-depth check to make sure the cascade will work in the planner
        trainInputs = os.listdir( os.path.join(nnUNet_raw, self.dataset_name, "imagesTr") )
        trainInputs = [file for file in trainInputs if "_0000" in file] # filter inputs to only grab channel 0
        testInputs = os.listdir( os.path.join(nnUNet_raw, self.dataset_name, "imagesTs") )
        testInputs = [file for file in testInputs if "_0000" in file]
        trainLabels = os.listdir( os.path.join(nnUNet_raw, self.dataset_name, "labelsTr") )
        testLabels = os.listdir( os.path.join(nnUNet_raw, self.dataset_name, "labelsTs") )

        if len(trainInputs) != len(trainLabels) or len(testInputs) != len(testLabels):
            print("It appears that the train and test sets for networks in the cascade are different.")
            print("Make sure you have the same train and test set for each network, then try again.")
            raise ImportError("Mismatch in number of train/test inputs and outputs")
        else:
            print("New dataset appears valid, but double check that the train and test sets are the same for all networks in cascade!")
        
        # generate .json for new dataset
        new_dataset = {}
        first_net_dataset = load_json( os.path.join(nnUNet_raw, self.cascade_dataset_names[0], "dataset.json") )
        last_net_datset = load_json( os.path.join(nnUNet_raw, self.cascade_dataset_names[-1], "dataset.json") )
        new_dataset["channel_names"] = first_net_dataset["channel_names"]
        new_dataset["labels"] = last_net_datset["labels"]
        new_dataset["numTraining"] = first_net_dataset["numTraining"]
        new_dataset["file_ending"] = new_dataset["file_ending"]
        # save .json for new dataset
        save_json(new_dataset, os.path.join(nnUNet_raw, self.dataset_name, "dataset.json") )
    
    
    
    # we override the run class, as there is some extra stuff we want to do with this extractor (mainly set up the new dataset from the existing datasets)
    def run(self, overwrite_existing: bool = False) -> dict:
        # we assume we can write to nnUnet_raw
        # if not, you'll need to create the dataset yourself
        if os.path.isdir( os.path.join(nnUNet_raw, self.dataset_name) ):
            print("The dataset directory already exists -- we are assuming this is intentional and will not remake the dataset.")
            print("Please check that the dataset given is correct and should exist.")
            
        else:
            self.gen_cascade_dataset
        
        # do some additional assignment now that the dataset should be made
        self.dataset_json = load_json(join(self.input_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.input_folder, self.dataset_json)

        # everything from here on out is the same as the run function in the original dataset function -- so we call the original run function
        fingerprint = super().run(overwrite_existing=overwrite_existing)
        return fingerprint


if __name__ == '__main__':
    dfe = cascadeDatasetFingerprintExtractor(2, 8)
    dfe.run(overwrite_existing=False)
