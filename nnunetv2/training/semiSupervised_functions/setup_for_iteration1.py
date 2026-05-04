import os
import shutil
import fire

# this function sets up the files to start the first iteration of semi-supervised nnUnet training
# basically we just move everything from our original training on just labeled data to dataset/iterations/mixed_dataset_000
def setup_iteration1(nnUnet_raw_dataset_path: str, nnUnet_preprocessed_dataset_path: str, nnUnet_results_dataset_path: str, fold: int):
    # create folders in dataset
    if not os.path.isdir( os.path.join(nnUnet_raw_dataset_path, "iterations") ):
        os.mkdir( os.path.join(nnUnet_raw_dataset_path, "iterations") )
    if not os.path.isdir( os.path.join(nnUnet_preprocessed_dataset_path, "iterations") ):
        os.mkdir( os.path.join(nnUnet_preprocessed_dataset_path, "iterations") )
    if not os.path.isdir( os.path.join(nnUnet_results_dataset_path, "iterations") ):
        os.mkdir( os.path.join(nnUnet_results_dataset_path, "iterations") )
        
    # setup paths and create folders for fold in preprocessed and raw
    rawPath = os.path.join(nnUnet_raw_dataset_path, "iterations", "fold_"+str(fold))
    ppPath = os.path.join(nnUnet_preprocessed_dataset_path, "iterations", "fold_"+str(fold))
    resultsPath = os.path.join(nnUnet_results_dataset_path, "iterations")
    
    if not os.path.isdir(rawPath):
        os.mkdir( rawPath )
    if not os.path.isdir(ppPath):
        os.mkdir( ppPath )
    
    # make folders for dataset_000 -- which just corresponds to our intial training on just labeled data
    dataset_raw = os.path.join(rawPath, "Dataset_mixed_001")
    dataset_pp = os.path.join(ppPath, "Dataset_mixed_001")
    dataset_res = os.path.join(resultsPath, "Dataset_mixed_001")
    
    # copy original everything from labeled training into _000 folders
    if not os.path.isdir(dataset_raw):
        print("copying raw files")
        shutil.copytree(nnUnet_raw_dataset_path, dataset_raw, ignore=shutil.ignore_patterns("iterations"))
    if not os.path.isdir(dataset_pp):
        print("copying preprocessed files")
        shutil.copytree(nnUnet_preprocessed_dataset_path, dataset_pp, ignore=shutil.ignore_patterns("iterations"))
    
    # need to check on results folder -- as this one is shared between folds
    if not os.path.isdir(dataset_res):
        print("copying results files")
        shutil.copytree(nnUnet_results_dataset_path, dataset_res, 
                        ignore=shutil.ignore_patterns("iterations", "train", "validation", "*Plots", "*.xlsx", "tstInference", "*.txt", "*.png"))
    
    # here we also set up our temporary directory for storing predictions on unlabeled data after each fold
    if not os.path.isdir( os.path.join(nnUnet_results_dataset_path, "tmp_outputs") ):
        os.mkdir( os.path.join(nnUnet_results_dataset_path, "tmp_outputs") )
    if not os.path.isdir( os.path.join(nnUnet_results_dataset_path, "tmp_outputs", "fold_"+str(fold)) ):
        os.mkdir( os.path.join(nnUnet_results_dataset_path, "tmp_outputs", "fold_"+str(fold)) )
    
if __name__ == "__main__":
    '''
    raw = "D:\\DeepLearningData\\nn_Unet_test\\nnUNet_raw\\Dataset031_cascadeFineTuning"
    pp = "D:\\DeepLearningData\\nn_Unet_test\\nnUNet_preprocessed\\Dataset031_cascadeFineTuning"
    results = "D:\\DeepLearningData\\nn_Unet_test\\nnUNet_results\\Dataset031_cascadeFineTuning"
    setup_iteration1(raw, pp, results, 0)
    '''
    fire.Fire(setup_iteration1)
        