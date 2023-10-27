import argparse
import glob
import os
import shutil

import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

import util.config as config
import util.util_classifier as util_classifier
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

# download data if it is not available
if not os.path.isdir(sourceDir0):
    filename = (
            "device_data_VICTREPhantoms_spic_"
            + LESIONDENSITY
            + "/"
            + DOSE.replace("+", "")
            + "/"
            + DENSITY
            + "/2/"
            + SIZE
            + "/"
            + DETECTOR
            + ".zip"
    )

    # Download dataset from huggingface
    print("downloading data from huggingface...")
    print("saving to " + str(config.dir_training_data))
    hf_hub_download(
        repo_id="didsr/msynth",
        use_auth_token=True,
        repo_type="dataset",
        local_dir=config.dir_training_data,  # download directory for this dataset
        local_dir_use_symlinks=False,
        filename='data/' + filename,
    )
    # Extract
    print("unzipping...")
    filenameZip = config.dir_training_data + "/data/" + filename
    shutil.unpack_archive(filenameZip, os.path.dirname(filenameZip).replace('data//data', 'data'), "zip")
    print("removing zip archive..")
    os.remove(filenameZip)
    shutil.rmtree(config.dir_training_data + 'data/')  # remove data/data/* folders due to huggingface dep

# download metadata if not available
if not os.path.isdir(config.dir_training_data + "metadata/bounds/"):
    filename = "metadata/bounds.zip"
    # Download metadata from huggingface
    print("downloading metadata from huggingface...")
    print("saving to " + config.dir_training_data)
    hf_hub_download(
        repo_id="didsr/msynth",
        use_auth_token=True,
        repo_type="dataset",
        local_dir=config.dir_training_data,  # download directory for this dataset
        local_dir_use_symlinks=False,
        filename='data/' + filename,
    )
    print("unzipping...")
    filenameZip = config.dir_training_data + "/data/" + "/metadata/bounds.zip"
    shutil.unpack_archive(filenameZip, os.path.dirname(filenameZip).replace('data//data', 'data'), "zip")
    print("removing zip archive..")
    os.remove(filenameZip)
    shutil.rmtree(config.dir_training_data + 'data/')  # remove data/data/* folders due to huggingface dep

l_detectorTypes = [DETECTOR]

preprocess_images = True
split_train_test = True

train_classifier = args.train

if DENSITY == "ALLDENSITIES":
    l_breastType = ["dense", "hetero", "scattered", "fatty"]
else:
    l_breastType = [DENSITY]

# image preprocessing
print("preprocessing images...")
if preprocess_images and not os.path.isfile(saveDir00 + "/annotations.txt"):
    saveDir = saveDir00
    annotationsFile = saveDir + "/annotations.txt"
    saveDir_images = saveDir + "/" + "images/"
    os.makedirs(saveDir_images, exist_ok=True)
    annotations_list = []
    examples = []
    for detectorType in l_detectorTypes:
        if SIZE == "ALLSIZES":
            print("All sizes")
            examples = glob.glob(
                sourceDir0.replace("ALLSIZES", "*") + "/*/*/projection_DM*.raw"
            )

        elif LESIONDENSITY == "ALLLESIONDENSITY":
            examples = glob.glob(
                sourceDir0.replace("ALLLESIONDENSITY", "*") + "/*/*/projection_DM*.raw"
            )

        elif DENSITY == "ALLDENSITIES":
            print("All breast types")
            examples_temp = []
            l_breastTypeDose = ["1.73e+09", "1.02e+10", "2.04e+10", "2.22e10"]
            for breastType, dose in list(zip(l_breastType, l_breastTypeDose)):
                sourceDir0_temp = sourceDir0.replace(DOSE, dose)
                print("sourceDir0_temp:", sourceDir0_temp)
                sourceDir0_temp2 = sourceDir0_temp.replace("ALLDENSITIES", breastType)
                print("sourceDir0_temp2", sourceDir0_temp2)
                examples_temp.append(
                    glob.glob(
                        sourceDir0_temp.replace("ALLDENSITIES", breastType)
                        + "*/*/projection_DM*.raw"
                    )
                )
                examples = [example for item in examples_temp for example in item]
        else:
            examples = glob.glob(sourceDir0 + "/*/*/projection_DM*.raw")

        for example in tqdm(examples):
            saveName = (
                example.replace(sourceDir00, "").replace("/", "_").replace(".raw", "")
            )
            file_outputProjection = (
                    os.path.dirname(example) + "/" + "output_projection.out"
            )
            stringToSearch = "adipose"
            dose = float(DOSE)
            filename_mhd = example.replace(".raw", ".mhd")

            lesion_present = util_preprocessing.get_lesion_label(filename_mhd)
            if lesion_present > 0:
                label = "lesion"
            else:
                label = "nolesion"

            # read image
            savePath = example.replace(sourceDir00, saveDir00).replace(".raw", ".png")
            tmp = util_preprocessing.preprocess_raw_file(filename_mhd)
            np.save(saveDir_images + "/" + saveName + ".npy", tmp)

            annotation = (
                    saveName
                    + ".npy"
                    + " "
                    + str(DENSITY)
                    + " "
                    + str(detectorType)
                    + " "
                    + str(dose)
                    + " "
                    + label
            )
            annotations_list.append(annotation)

    with open(annotationsFile, "w") as f:
        for line in annotations_list:
            f.write(f"{line}\n")

# split into train/val/test
print("performing train/val/test split ...")
if split_train_test and not os.path.isdir(saveDir00[:-1] + "_basic_train_val_test/"):
    dirPath = saveDir00
    saveDir_trainValTest = dirPath[:-1] + "_basic_train_val_test/"

    flImages = open(dirPath + "annotations.txt", "r")
    lines = flImages.readlines()
    flImages.close()

    if SIZE == "ALLSIZES":
        print("ALLSIZES")
        lines_P_t = []
        lines_P = []
        lines_notP = []
        for lesion_size in list(["_5.0_", "_7.0_", "_9.0_"]):
            temp_lines = [line for line in lines if lesion_size in line]
            for i in range(len(temp_lines)):
                if i < 200 / 3:
                    lines_notP.append(temp_lines[i])
                if i >= 201 / 3 and i < 250 / 3:
                    lines_P.append(temp_lines[i])
                if i >= 250 / 3 and i < 1 / 3 * len(temp_lines):
                    lines_P_t.append(temp_lines[i])

    elif LESIONDENSITY == "ALLLESIONDENSITY":
        print("ALLDENSITIES")
        lines_P_t = []
        lines_P = []
        lines_notP = []
        for lesion_DENSITY in list(["_1.0_", "_1.06_", "_1.1_"]):
            temp_lines = [line for line in lines if lesion_DENSITY in line]
            print(len(temp_lines))
            for i in range(len(temp_lines)):
                if i < 200 / 3:
                    lines_notP.append(temp_lines[i])
                if i >= 201 / 3 and i < 250 / 3:
                    lines_P.append(temp_lines[i])
                if i >= 250 / 3 and i < 1 / 3 * len(temp_lines):
                    lines_P_t.append(temp_lines[i])

    elif DENSITY == "ALLDENSITIES":
        print("ALLDENSITIES")
        lines_P_t = []
        lines_P = []
        lines_notP = []
        for density in list(["dense", "hetero", "scattered", "fatty"]):
            temp_lines = [line for line in lines if density in line]
            for i in range(len(temp_lines)):
                if i < 200 / 4:
                    lines_notP.append(temp_lines[i])
                if i >= 200 / 4 and i < 250 / 4:
                    lines_P.append(temp_lines[i])
                if i >= 250 / 4 and i < 1 / 4 * len(temp_lines):
                    lines_P_t.append(temp_lines[i])

    else:
        lines = [line.strip() for line in lines]
        lines_notP = [lines[i] for i in range(len(lines)) if i < 200]
        lines_P = [lines[i] for i in range(len(lines)) if (i >= 200 and i < 250)]
        lines_P_t = [lines[i] for i in range(len(lines)) if i >= 250]

    x_train = lines_notP
    x_val = lines_P
    x_test = lines_P_t

    os.makedirs(saveDir_trainValTest, exist_ok=True)
    os.makedirs(saveDir_trainValTest + "/train/withlesion/", exist_ok=True)
    os.makedirs(saveDir_trainValTest + "/train/nolesion/", exist_ok=True)
    os.makedirs(saveDir_trainValTest + "/val/withlesion/", exist_ok=True)
    os.makedirs(saveDir_trainValTest + "/val/nolesion/", exist_ok=True)
    os.makedirs(saveDir_trainValTest + "/test/withlesion/", exist_ok=True)
    os.makedirs(saveDir_trainValTest + "/test/nolesion/", exist_ok=True)

    for i in tqdm(range(len(x_train))):
        line = x_train[i]
        imgname = line.split(" ")[0]
        if not os.path.isfile(dirPath + "images/" + imgname):
            print("PROBLEM")
        if "nolesion" in line:
            shutil.copyfile(
                dirPath + "images/" + imgname,
                saveDir_trainValTest + "/train/nolesion/" + imgname,
            )
        else:
            shutil.copyfile(
                dirPath + "images/" + imgname,
                saveDir_trainValTest + "/train/withlesion/" + imgname,
            )

    for i in tqdm(range(len(x_val))):
        line = x_val[i]
        imgname = line.split(" ")[0]
        if not os.path.isfile(dirPath + "images/" + imgname):
            print("PROBLEM")
        if "nolesion" in line:
            shutil.copyfile(
                dirPath + "images/" + imgname,
                saveDir_trainValTest + "/val/nolesion/" + imgname,
            )
        else:
            shutil.copyfile(
                dirPath + "images/" + imgname,
                saveDir_trainValTest + "/val/withlesion/" + imgname,
            )

    for i in tqdm(range(len(x_test))):
        line = x_test[i]
        imgname = line.split(" ")[0]
        if not os.path.isfile(dirPath + "images/" + imgname):
            print("PROBLEM")
        if "nolesion" in line:
            shutil.copyfile(
                dirPath + "images/" + imgname,
                saveDir_trainValTest + "/test/nolesion/" + imgname,
            )
        else:
            shutil.copyfile(
                dirPath + "images/" + imgname,
                saveDir_trainValTest + "/test/withlesion/" + imgname,
            )

# train classifier
if train_classifier:
    dirPath = saveDir00
    saveDir_trainValTest = dirPath[:-1] + "_basic_train_val_test/"

    util_classifier.train_model(saveDir_trainValTest, nickname, nreps=args.ntrainreps)
