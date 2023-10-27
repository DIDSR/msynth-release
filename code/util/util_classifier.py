import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import util.config as config
import util.util as util
from PIL import Image
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import DatasetFolder

batch_size = 64
num_workers = 12


class AddGaussianNoise(object):
    # https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745/2
    def __init__(self, mean=0.0, std=1.0, prob=0.5):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if random.random() < 0.5:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            util.globalNormalize(),  # transforms.Normalize(0., 1.),
            transforms.RandomAffine(30),
            transforms.RandomHorizontalFlip(),
            AddGaussianNoise()
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            util.globalNormalize(),  # transforms.Normalize(0., 1.),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            util.globalNormalize(),  # transforms.Normalize(0., 1.),
        ]
    ),
}


def load_image(impath):
    if config.FLOAT:
        tmp = np.load(impath)  # .astype(np.uint8)
    else:
        tmp = np.load(impath).astype(np.uint8)
    tmp = Image.fromarray(tmp)
    return tmp


class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3, n_classes=107):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

        self.model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=n_classes, in_chans=1
        )
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def configure_optimizers(self):
        opt = optim.RMSprop(self.model.parameters(), self.lr)
        return opt

    def training_step(self, dl, idx):
        x, targets = dl
        inputs = self.model(x)
        inputs_sigm = self.sigmoid(inputs)
        loss = self.loss(inputs, targets.unsqueeze(1).float())

        accu = (
                (torch.round(inputs_sigm.flatten()) == targets).sum() / x.shape[0]
        ).item()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accu, prog_bar=True)

        return {"loss": loss, "accu": accu}

    def training_epoch_end(self, outputs) -> None:
        loss_sum = 0
        accu_sum = 0
        for output in outputs:
            loss_sum += output["loss"].item()
            accu_sum += output["accu"]
        print("training epoch loss ", loss_sum / len(outputs))
        print("training epoch accu ", accu_sum / len(outputs))

    def validation_step(self, dl, idx):
        x, targets = dl
        inputs = self.model(x)
        inputs_sigm = self.sigmoid(inputs)
        loss = self.loss(inputs, targets.unsqueeze(1).float())
        accu = (
                (torch.round(inputs_sigm.flatten()) == targets).sum() / x.shape[0]
        ).item()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accu, prog_bar=True)

        return {"loss": loss, "accu": accu}

    def validation_epoch_end(self, outputs) -> None:
        loss_sum = 0
        accu_sum = 0
        for output in outputs:
            loss_sum += output["loss"]
            accu_sum += output["accu"]
        print("validation epoch loss ", loss_sum.item() / len(outputs))
        print("validation epoch accu ", accu_sum / len(outputs))
        self.log("val_epoch_loss", loss_sum.item() / len(outputs))
        self.log("val_epoch_acc", accu_sum / len(outputs))


def train_model(data_path, nickname, nreps=10, nepochs=200):
    print("training model " + str(nreps) + "times")
    if config.FLOAT:
        training_dir = data_path + "/float/"
    else:
        training_dir = data_path + "/uint8/"
    print("training_dir ", training_dir)
    os.makedirs(training_dir, exist_ok=True)
    print("Log file will be saved in " + training_dir + "/" + "train.log")
    log_file = open(training_dir + "/" + "train.log", "a")

    sys.stdout = log_file
    sys.stderr = log_file

    l_to_train = []
    for k in range(nreps):
        l_to_train.append([data_path, nickname])
    for trialEx in l_to_train:
        print("TRIAL " + str(trialEx))

        data_dir = trialEx[0]
        modelNickame = trialEx[1]
        print("data_dir " + data_dir)
        print("modelNickame " + modelNickame)

        image_datasets = {
            x: DatasetFolder(
                root=os.path.join(data_dir, x),
                transform=data_transforms[x],
                loader=load_image,
                extensions=(".npy"),
            )
            for x in ["train", "val", "test"]
        }

        data_loader = {
            x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            for x in ["train", "val", "test"]
        }

        train_dl = data_loader["train"]
        val_dl = data_loader["val"]
        data_loader["test"]

        model = Classifier(n_classes=1, lr=1e-4)
        checkpoint_callback = ModelCheckpoint(
            verbose=True,
            save_top_k=1,
            monitor="val_acc",
            mode="max",
            every_n_epochs=1,
            filename="efficientnetb0"
                     + modelNickame
                     + "_{epoch}-{val_acc:.2f}-{train_acc:.2f}",
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=nepochs,
            callbacks=[TQDMProgressBar(refresh_rate=5), checkpoint_callback],
            default_root_dir=training_dir,
        )
        trainer.fit(model, train_dl, val_dl)
        print(checkpoint_callback.best_model_path)
        print(
            "Restoring states from the checkpoint path at "
            + checkpoint_callback.best_model_path
        )
        trainer.validate(
            model,
            ckpt_path=checkpoint_callback.best_model_path,
            dataloaders=val_dl,
            verbose=True,
        )
    print("done model training")
