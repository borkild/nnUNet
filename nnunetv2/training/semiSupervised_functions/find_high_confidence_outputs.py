import os
import numpy as np
import fire

def find_high_confidence_outputs(target_folder: str, txtFileSavePath: str, background_channel:int =0, file_extension:str =".npz", confidence_threshold:float = 0.90):
    # get list of files in target folder
    output_files = os.listdir(target_folder)
    
    # iterate through files, checking max value for each channel, outside of the background channel
    # NOTE: we assume that there should be a positive instance of a non-background channel in every scan
    # if that is not the case, you will likely want to implement your own version of this function
    high_confidence_outputs = []
    for curScan in output_files:
        if file_extension in curScan:
            output_dict = np.load( os.path.join(target_folder, curScan) ) # for now I'm assuming we use .npz format, as that is the default for nnUnet
            output = output_dict['probabilities']
            noBG_output = np.delete(output, background_channel, axis=0)
            max_vals = np.max(noBG_output, axis=(1,2,3))
            if np.sum( max_vals > confidence_threshold ) == noBG_output.shape[0]:
                high_confidence_outputs.append( os.path.join(target_folder, curScan) )
    # write outputs to text file
    writeListToTextFile(high_confidence_outputs, txtFileSavePath)
    
def writeListToTextFile(listToWrite: list, saveFilePath: str):
    with open(saveFilePath, 'w') as file:
        for curScan in listToWrite:
            file.write(curScan + "\n")
            
    

    
if __name__ == "__main__":
    fire.Fire(find_high_confidence_outputs)

