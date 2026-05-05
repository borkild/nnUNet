import os
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.semiSupervised_functions.generate_mixed_dataset import load_txt_file
import fire


# we manually update the split file -- this way the unlabeled data stays only in the training set
def update_split_file(current_preprocessed_path: str, original_preprocessed_path: str, unlabeled_list_path: str, cur_fold: int, cur_iteration: int):
    original_split = load_json( join(original_preprocessed_path, "splits_final.json") )
    # load in list of unlabeled scans
    unlabeled_list = load_txt_file(unlabeled_list_path)
    unlabeled_IDs = []
    # strip list to have only scan ID
    for curScan in unlabeled_list:
        unlabeled_IDs.append( strip_fullPath_to_ID(curScan) )
    # now combine original split with unlabled data for this fold
    original_split[cur_fold]['train'] = original_split[cur_fold]['train'] + unlabeled_IDs
    # save out json
    final_save_path = join(current_preprocessed_path, "Dataset"+str(cur_iteration).zfill(3) + "_mixed", "splits_final.json")
    save_json(original_split, final_save_path)
    


def strip_fullPath_to_ID(fullPath: str):
    tmp_path = fullPath.strip(".")
    if os.name == "nt":
        all_dirs = tmp_path[0].strip("\\")
    else:
        all_dirs = tmp_path[0].strip("/")
    ID = all_dirs[-1]
    return ID
    

if __name__ == "__main__":
    fire.Fire(update_split_file)