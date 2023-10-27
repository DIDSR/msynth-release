import argparse
import glob
import os

import util.config as config
import util.util_preprocessing as util_preprocessing
import util.util_testing as util_testing

parser = argparse.ArgumentParser()
# Training parameters
parser.add_argument("--density", type=str, help="Breast Density")
parser.add_argument("--size", type=str, help="Lesion Size")
parser.add_argument("--lesiondensity", type=str, help="Lesion Density")
parser.add_argument("--detector", type=str, help="Detector")
parser.add_argument("--dose", type=str, help="Dose")
# Testing parameters
parser.add_argument("--density_test", type=str, help="Test Breast Density")
parser.add_argument("--size_test", type=str, help="Test Lesion Size")
parser.add_argument("--lesiondensity_test", type=str, help="Test Lesion Density")
parser.add_argument("--dose_test", type=str, help="Test Dose")
args = parser.parse_args()

args.dose = args.dose.replace("+", "")  # for compatibility
args.dose_test = args.dose_test.replace("+", "")  # for compatibility

DENSITY = args.density
SIZE = args.size
LESIONDENSITY = args.lesiondensity
DETECTOR = args.detector
DOSE = args.dose

DENSITY_TEST = args.density_test
SIZE_TEST = args.size_test
LESIONDENSITY_TEST = args.lesiondensity_test
DOSE_TEST = args.dose_test
DETECTOR_TEST = "SIM"

sourceDir00, sourceDir0 = util_preprocessing.get_source_dirs(
    config.dir_training_data, LESIONDENSITY, DENSITY, SIZE, DETECTOR, DOSE
)
saveDir00 = util_preprocessing.get_save_dir(
    config.dir_training_data_preprocessed, DENSITY, SIZE, LESIONDENSITY, DOSE, DETECTOR
)
nickname = util_preprocessing.get_model_nickname(
    DENSITY, SIZE, DETECTOR, LESIONDENSITY, DOSE
)

create_evaluations_dicts = True

dirPath = saveDir00
saveDir_trainValTest = dirPath[:-1] + "_basic_train_val_test/"

# get the example ID for the log in test data for processing
test_log_list = []
test_log_ID_list = []
testDataDir_nolesion = (
        config.dir_training_data_preprocessed
        + "/device_data_VICTREPhantoms_"
        + DENSITY_TEST
        + "_spic"
        + SIZE_TEST
        + "_id2_"
        + DETECTOR_TEST
        + "_"
        + LESIONDENSITY_TEST
        + "_"
        + DOSE_TEST
        + "_preprocessed1_basic_train_val_test/test/nolesion"
)
testDataDir_withlesion = (
        config.dir_training_data_preprocessed
        + "/device_data_VICTREPhantoms_"
        + DENSITY_TEST
        + "_spic"
        + SIZE_TEST
        + "_id2_"
        + DETECTOR_TEST
        + "_"
        + LESIONDENSITY_TEST
        + "_"
        + DOSE_TEST
        + "_preprocessed1_basic_train_val_test/test/withlesion"
)
for log_name in os.listdir(testDataDir_nolesion):
    test_log_list.append(log_name)
for log_name in os.listdir(testDataDir_withlesion):
    test_log_list.append(log_name)
for item in test_log_list:
    test_log_ID_list.append(item.split("projection_DM")[1].split(".npy")[0])

# generate dictionaries
if create_evaluations_dicts:
    dirPath = saveDir00
    saveDir_trainValTest = dirPath[:-1] + "_basic_train_val_test/"

    if config.FLOAT:
        model_run_file = (
                saveDir00[:-1] + "_basic_train_val_test/" + "/float/" + "/" + "train.log"
        )
    else:
        model_run_file = (
                saveDir00[:-1] + "_basic_train_val_test/" + "/uint8/" + "/" + "train.log"
        )

    model_runs = [model_run_file]
    dict_dir = saveDir00[:-1] + "_basic_train_val_test/" + "/" + "dicts/all/"
    os.makedirs(dict_dir, exist_ok=True)
    util_testing.run_dict_script(
        test_log_list[0:1],
        test_log_ID_list,
        model_runs,
        dict_dir,
        logDir="log_victre/",
        path_examples=glob.glob(testDataDir_nolesion + "/*") + glob.glob(testDataDir_withlesion + "/*"),
        NEXAMPLES=len(test_log_ID_list),
    )
