import os
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
import fire


def update_and_save_plan(cur_iteration: int, preprocessed_path: str):
    old_plan_path = os.path.join(preprocessed_path, "Dataset" + str(cur_iteration-1).zfill(3) + "_mixed", "nnUNetCascadePlans.json")
    plan = load_json(old_plan_path)
    # update dataset name
    plan["dataset_name"] = "Dataset" + str(cur_iteration).zfill(3) + "_mixed"
    save_json(plan, os.path.join(preprocessed_path, "Dataset" + str(cur_iteration).zfill(3) + "_mixed", "nnUNetCascadePlans.json"))


if __name__ == "__main__":
    fire.Fire(update_and_save_plan)