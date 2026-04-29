import os
import numpy as np
import nrrd
import shutil
import fire


def generate_mixed_label_dataset(current_iteration: int, unlabeled_txt_file_list: str, overall_dataset_path: str, output_file_format: str = ".nrrd"):
    new_dataset_loc = os.path.join(overall_dataset_path, "iterations", "mixed_dataset_" + str(current_iteration+1).zfill(3) )
    nd_image_path = os.path.join(new_dataset_loc, "imagesTr")
    nd_label_path = os.path.join(new_dataset_loc, "labelsTr")
    unlabeled_folder_path = os.path.join(overall_dataset_path, "unlabeledimagesTr")
    # create folder to store new mixed label dataset 
    if not os.path.isdir( new_dataset_loc ):
        os.mkdir( new_dataset_loc )
        os.mkdir( nd_image_path )
        os.mkdir( nd_label_path )
    else:
        raise FileExistsError("Folder for next iteration already exists -- something is up -- double check your output directories")
    
    # move labeled training samples from dataset to the next iteration's dataset folder
    shutil.copytree( os.path.join(overall_dataset_path, "imagesTr"), nd_image_path, dirs_exist_ok=True )
    shutil.copytree( os.path.join(overall_dataset_path, "labelsTr"), nd_label_path, dirs_exist_ok=True )
    
    # load text file of scans in as list
    unlabeled_scans = load_txt_file(unlabeled_txt_file_list)
    
    # save scans out in required format (stuck with .nrrd for now)
    for curScan in unlabeled_scans:
        curPath = os.path.join(unlabeled_folder_path, curScan)
        id = curScan.split(".")
        id = id[0].split("/")
        outPath = os.path.join(nd_label_path, id[-1]+output_file_format)
        # copy image
        shutil.copy( os.path.join(unlabeled_folder_path, id[-1] + output_file_format), nd_image_path )
        # copy label
        save_npz_as_nrrd(curPath, outPath)
    
    
    


# function to load in scans from txt file
def load_txt_file(txt_file_path: str):
    with open(txt_file_path, 'r') as file:
        txt_file_content = file.readlines()
        
    return txt_file_content

def save_npz_as_nrrd(npz_path: str, label_path_to_write: str):
    # load in npz with prediction data
    prediction_data = np.load(npz_path)
    prediction = prediction_data["probabilities"]
    # for now we assume the background is the first dimension
    noBG_pred = np.delete(prediction, 0, axis=0)
    # load in nrrd as well, this way we can get spacing and origin for the nrrd file we write
    basic_path = npz_path.split(".")
    _, nrrdHeader = nrrd.read(basic_path[0] + ".nrrd")
    outputHeader = {}
    # check for origin and spacing fields
    if "space directions" in nrrdHeader:
        outputHeader["space directions"] = nrrdHeader["space directions"]
    elif "spacings" in nrrdHeader:
        outputHeader["spacings"] = nrrdHeader["spacings"]
    
    if "space origin" in nrrdHeader:
        outputHeader["space origin"] = nrrdHeader["space origin"]
    
    print("Writing: " + label_path_to_write)
    nrrd.write(label_path_to_write, noBG_pred, outputHeader)
    
    
    
    
if __name__ == "__main__":
    fire.Fire(generate_mixed_label_dataset)
    