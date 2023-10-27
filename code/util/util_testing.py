import time

import numpy as np
import torch
import util.config as config
import util.util_classifier as util_classifier
from PIL import Image
from sklearn.metrics import classification_report
from tqdm import tqdm
from util.config import *
from util.config import *


def evaluate_models_on_log(
        model_names,
        FLNAME,
        test_log_ID_list,
        DEBUG,
        NEXAMPLES=100,
        dict_dir="",
        path_examples="",
        logDir="log/",
):
    l_save_dict_name = []
    for best_path in model_names:
        print('Checkpoint: '+best_path)
        dd_name = get_dict_names([best_path], FLNAME, dict_dir)
        model = util_classifier.Classifier.load_from_checkpoint(best_path)
        model.eval()
        if CUDA:
            model.cuda()

        start_time = time.time()
        l_ba = []
        l_outputs = []
        i = 0
        for i in range(1):
            out = FLNAME
            (
                true_labels0,
                predicted_labels0,
                prob_y0,
                l_images0,
                l_lesion_masks,
                _,
                num_examples,
            ) = process_output(
                out,
                model,
                example_IDs=test_log_ID_list,
                DOSEID=i,
                logDir=logDir,
                path_examples=path_examples,
            )
            balanced_accuracy = get_ba(
                true_labels0, predicted_labels0
            )  # calculate balanced accuracy
            print("Balanced Accuracy (BA): %.3f" %(balanced_accuracy))
            l_ba.append(balanced_accuracy)
            doseval = float(
                path_examples[0].split("/")[-1].split("_")[0]
            )

            d_output = {
                "true_labels": true_labels0,
                "predicted_labels0": predicted_labels0,
                "prob_y0": prob_y0,
                "l_images0": [x.numpy() for x in l_images0],
                "l_lesion_masks": l_lesion_masks,
                "out": out,
                "balanced_accuracy": balanced_accuracy,
                "dose": doseval,
                "mean_time": [],
                "num_examples": num_examples,
            }
            l_outputs.append(d_output)
            i += 1
        print("Evaluation Time: %.3f" %(time.time() - start_time) + '\n')
        save_dict_name = save_dict(l_outputs, best_path, FLNAME, NEXAMPLES, dict_dir)
        l_save_dict_name.append(save_dict_name)
    return l_save_dict_name


def process_output(
        out,
        classifier_model,
        example_IDs,
        num_examples=50,
        DOSEID=1,
        logDir="log/",
        path_examples="",
):
    ############################################################
    ## get accuracy
    l_raw_images0 = []
    l_raw_images = []
    true_labels = []
    l_lesion_masks = []
    for exampleID in example_IDs:
        if path_examples != "":
            save_fileName = [ex for ex in path_examples if "." + str(exampleID) in ex][
                0
            ]
            tmp = np.load(save_fileName)
            tmp = Image.fromarray(tmp)
            l_raw_images.append(tmp)
            if "withlesion" in save_fileName:
                lesion_present = 1.0
            else:
                lesion_present = 0.0
            true_labels.append(lesion_present)
        else:
            raise Exception("Please specify path_examples!")

    # load images and run through network
    l_images = []
    transform = util_classifier.data_transforms["test"]
    img_id = 0
    for raw_img in l_raw_images:
        image_data = transform(raw_img)
        l_images.append(image_data)
        img_id += 1

    tens_images = torch.stack(l_images, dim=0)
    if CUDA:
        tens_images = tens_images.cuda()
    classifier_model.model.eval()
    output = classifier_model.model(tens_images)  # .float())
    prob_y_t = classifier_model.sigmoid(output)
    output_sigm_round = torch.round(prob_y_t)
    if CUDA:
        prob_y = prob_y_t.cpu().detach().numpy()[:, 0]
    else:
        prob_y = prob_y_t.detach().numpy()[:, 0]
    predicted_labels = output_sigm_round.data.cpu().numpy().flatten().tolist()

    return (
        true_labels,
        predicted_labels,
        prob_y,
        l_images,
        l_lesion_masks,
        l_raw_images0,
        num_examples,
    )


def get_ba(true_labels0, predicted_labels0):
    report = classification_report(true_labels0, predicted_labels0, output_dict=True)
    sensitivity = report["1.0"]["recall"]
    specificity = report["0.0"]["recall"]
    balanced_accuracy = (sensitivity + specificity) * 0.5
    return balanced_accuracy


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


def save_dict(l_outputs, mpath, FLNAME, NEXAMPLES, dict_dir):
    l_outputs_to_save = []
    for dic in l_outputs:
        dic1 = {
            "dose": dic["dose"],
            "balanced_accuracy": dic["balanced_accuracy"],
            "true_labels": dic["true_labels"],
            "predicted_labels0": dic["predicted_labels0"],
            "prob_y0": dic["prob_y0"],
            "mpath": mpath,
            "FLNAME": FLNAME,
            "NEXAMPLES": NEXAMPLES,
        }
        l_outputs_to_save.append(dic1)
    saveName = get_saveDict_name_test(mpath, FLNAME, FLOAT, dict_dir)
    np.save(saveName, l_outputs_to_save)
    return saveName


def get_saveDict_name_test(mpath, FLNAME, FLOAT=False, dict_dir=""):
    # "tmp/dicts/"
    tmp = FLNAME.split("_")
    saveName = (
            dict_dir
            + tmp[5]
            + "_"
            + tmp[6]
            + "_"
            + tmp[4]
            + "_"
            + tmp[0]
            + "_"
            + tmp[7].split(".")[0]
            + "."
            + tmp[7].split(".")[1]
            + "__"
            + "__".join(mpath.replace(".ckpt", "").split("/")[-3:])
            + "__"
            + "FLOAT"
            + str(int(FLOAT))
            + ".npy"
    )
    return saveName


def get_dict_names(model_names, FLNAME, dict_dir):
    l_save_dict_name = []
    for mpath in model_names:
        saveName = get_saveDict_name_test(mpath, FLNAME, FLOAT=False, dict_dir=dict_dir)
        l_save_dict_name.append(saveName)
    return l_save_dict_name


def run_dict_script(
        l_FLNAME,
        test_log_ID_list,
        model_runs,
        dict_dir,
        logDir="log/",
        path_examples="",
        NEXAMPLES=100,
):
    model_names_all = []
    for model_runs_name in model_runs:
        flLog = open(model_runs_name, "r")
        lines = flLog.readlines()
        flLog.close()

        lines_model = [
            line for line in lines if "Restoring" in line and "efficientnetb0" in line
        ]
        model_names = [line_model.strip().split(" ")[-1] for line_model in lines_model]
        model_names_all = model_names_all + model_names

        for FLNAME in l_FLNAME:
            l_save_dict_name = evaluate_models_on_log(
                model_names_all,
                FLNAME,
                test_log_ID_list,
                DEBUG=config.DEBUG,
                dict_dir=dict_dir,
                path_examples=path_examples,
                NEXAMPLES=NEXAMPLES,
            )
