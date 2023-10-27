import argparse
import os
import shutil

from huggingface_hub import hf_hub_download

import util.config as config
import util.util_preprocessing as util_preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--density", type=str, help="Breast Density")
parser.add_argument("--size", type=str, help="Lesion Size")
parser.add_argument("--lesiondensity", type=str, help="Lesion Density")
parser.add_argument("--detector", type=str, help="Detector")
parser.add_argument("--dose", type=str, help="Dose")
parser.add_argument(
    "--ntrainreps", type=int, help="Number of times to train each model", default=10
)
parser.add_argument("--train", action="store_true", default=False)
args = parser.parse_args()

args.dose = args.dose.replace("+", "")  # for compatibility

DENSITY = args.density
SIZE = args.size
LESIONDENSITY = args.lesiondensity
DETECTOR = args.detector
DOSE = args.dose

sourceDir00, sourceDir0 = util_preprocessing.get_source_dirs(
    config.dir_training_data, LESIONDENSITY, DENSITY, SIZE, DETECTOR, DOSE
)
saveDir00 = util_preprocessing.get_save_dir(
    config.dir_training_data_preprocessed, DENSITY, SIZE, LESIONDENSITY, DOSE, DETECTOR
)
nickname = util_preprocessing.get_model_nickname(
    DENSITY, SIZE, DETECTOR, LESIONDENSITY, DOSE
)

dirPath = saveDir00
saveDir_trainValTest = dirPath[:-1] + "_basic_train_val_test/"

# download or copy float/ directory with models and train.log to saveDir_trainValTest from huggingface
if not os.path.isfile(saveDir_trainValTest + "/float/train.log"):
    subfolder = (
            "pretrained_models/device_data_VICTREPhantoms_"
            + DENSITY
            + "_spic"
            + SIZE
            + "_id2_SIM_"
            + LESIONDENSITY
            + "_"
            + DOSE.replace("+", "")
            + "_preprocessed1_basic_train_val_test/"
    )
    filename = "float.zip"
    # Download dataset from huggingface
    print("downloading pretrained models from huggingface...")
    print(
        "saving to " + saveDir_trainValTest
    )

    hf_hub_download(
        repo_id="didsr/msynth",
        use_auth_token=True,
        repo_type="dataset",
        local_dir=saveDir_trainValTest,  # download directory for this dataset
        local_dir_use_symlinks=False,
        filename='data/'+subfolder + filename,
    )

    # Extract
    print("unzipping...")
    filenameZip = (
            saveDir_trainValTest + 'data/'+ subfolder + filename
    )
    shutil.unpack_archive(filenameZip, saveDir_trainValTest, "zip")
    print("removing zip archive..")
    shutil.rmtree(saveDir_trainValTest + "data/pretrained_models/")

# adjust train.log to contrain correct paths
flOpen = open(saveDir_trainValTest + "/float/train_TRANSFER.log", "r")
lines = flOpen.readlines()
flOpen.close()

lines = [
    line.replace(
        "#PATHTOBEREPLACED#/training_data_preprocessed_new/",
        config.dir_training_data_preprocessed,
    )
    for line in lines
]

flOpen = open(saveDir_trainValTest + "/float/train.log", "w")
for line in lines:
    flOpen.write(line)
flOpen.close()
