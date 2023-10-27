import os

import numpy as np
import util.config as config
import util.util as util


def preprocess_raw_file(filename_mhd):
    filename_raw = filename_mhd.replace(".mhd", ".raw")
    data = util.read_mhd(filename_mhd)
    pixel_array = np.fromfile(filename_raw, dtype="float32").reshape(
        data["NDims"], data["DimSize"][1], data["DimSize"][0]
    )
    tmp = pixel_array[0]
    if "dense" in filename_mhd:
        bounds_name = "bounds_dense.npy"
    elif "fatty" in filename_mhd:
        bounds_name = "bounds_fatty.npy"
    elif "hetero" in filename_mhd:
        bounds_name = "bounds_hetero.npy"
    elif "scattered" in filename_mhd:
        bounds_name = "bounds_scattered.npy"
    else:
        raise Exception("Bounds not found!")

    bounds_saved = np.load(
        config.dir_training_data + "metadata/" + "bounds/" + bounds_name,
        allow_pickle=True,
    )
    tmp = tmp[bounds_saved[0], bounds_saved[1]]
    X = np.std(tmp) * 2
    tmp[tmp < np.mean(tmp) - X] = 0
    tmp[tmp > np.mean(tmp) + X] = X + np.mean(tmp)
    return tmp


def get_lesion_label(filename_mhd):
    locFile = filename_mhd.replace(".mhd", ".loc")
    if os.path.isfile(locFile):
        lesion_present = 1.0
    else:
        lesion_present = 0.0
    return lesion_present


def get_model_nickname(DENSITY, SIZE, DETECTOR, LESIONDENSITY, DOSE):
    nickname = (
            "out_victre_"
            + DENSITY
            + "_spic"
            + SIZE
            + "_id2_"
            + DETECTOR
            + "_"
            + LESIONDENSITY
            + "_"
            + DOSE
            + ".out"
    )
    return nickname


def get_source_dirs(dir_training_data, LESIONDENSITY, DENSITY, SIZE, DETECTOR, DOSE):
    sourceDir00 = (
            dir_training_data + "/device_data_VICTREPhantoms_spic_" + LESIONDENSITY + "/"
    )
    sourceDir0 = (
            sourceDir00 + DOSE + "/" + DENSITY + "/2/" + SIZE + "/" + DETECTOR + "/"
    )
    return sourceDir00, sourceDir0


def get_save_dir(
        dir_training_data_preprocessed, DENSITY, SIZE, LESIONDENSITY, DOSE, DETECTOR
):
    saveDir00 = dir_training_data_preprocessed + "/device_data_VICTREPhantoms_"
    saveDir00 += (
            DENSITY
            + "_spic"
            + SIZE
            + "_id2_"
            + DETECTOR
            + "_"
            + LESIONDENSITY
            + "_"
            + DOSE
            + "_preprocessed1/"
    )
    return saveDir00
