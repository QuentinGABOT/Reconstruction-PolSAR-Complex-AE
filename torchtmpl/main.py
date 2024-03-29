# coding: utf-8

# Standard imports
import logging
import sys
from os import path, makedirs
import pathlib
import random

# External imports
import yaml
import wandb
import torch
import math
import torch.nn as nn
import torchinfo.torchinfo as torchinfo
import torchcvnn.nn.modules as c_nn
from PIL import Image
import numpy as np

# Local imports
from . import data as dt
from . import models
from . import optim
from . import utils
import torchtmpl as tl
from torchtmpl.models import VAE, UNet, AutoEncoder


def init_weights(m):
    if (
        isinstance(m, nn.Linear)
        or isinstance(m, nn.Conv2d)
        or isinstance(m, nn.ConvTranspose2d)
    ):
        c_nn.init.complex_kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_path, config, device):
    model = models.build_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load(config):
    if config["pretrained"]["check"]:
        checkpoint_path = config["pretrained"]["path"]
        checkpoint = torch.load(checkpoint_path)
        seed = checkpoint["seed"]
        seed_everything(seed)
    else:
        seed = math.floor(random.random() * 10000)
        seed_everything(seed)

    cdtype = torch.complex64
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        if config["pretrained"]["check"]:
            wandb.init(
                project=wandb_config["project"],
                resume="must",
                id=checkpoint["wandb_id"],
            )
        else:
            wandb.init(project=wandb_config["project"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader = dt.get_dataloaders(data_config, use_cuda)

    # Build the model
    logging.info("= Model")
    model_config = config

    model = models.build_model(model_config)
    model.apply(init_weights)

    if config["pretrained"]["check"]:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = models.build_model(config)
        model.apply(init_weights)

    with torch.no_grad():
        model.eval()
        dummy_input = torch.zeros(
            (
                config["data"]["batch_size"],
                config["data"]["num_channels"],
                config["data"]["img_size"],
                config["data"]["img_size"],
            ),
            dtype=cdtype,
            requires_grad=False,
        )  # gérer la dimension d'entrée
        out_conv = model(dummy_input)

    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = tl.optim.get_loss(config["loss"]["name"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = tl.optim.get_optimizer(optim_config, model.parameters())

    epoch = 1

    if config["pretrained"]["check"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"] + 1  # Start from the next epoch

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["model"]["class"]

    if not path.isdir(logging_config["logdir"]):
        makedirs(logging_config["logdir"])

    if config["pretrained"]["check"]:
        logdir = checkpoint["logdir"]
    else:
        logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
        if not path.isdir(logdir):
            makedirs(logdir)

    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open("config.yml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    if next(iter(train_loader)).shape == 2:
        input_size = next(iter(train_loader))[0].shape
    else:
        input_size = next(iter(train_loader)).shape

    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size, dtypes=[cdtype])}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )

    with open(logdir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    return (
        model,
        optimizer,
        loss,
        train_loader,
        valid_loader,
        device,
        input_size,
        epoch,
        seed,
        wandb_log,
        logdir,
    )


def train(config):
    """
    data.delete_folders_with_few_pngs()
    print("Done")
    input()
    """

    (
        model,
        optimizer,
        loss,
        train_loader,
        valid_loader,
        device,
        input_size,
        epoch,
        seed,
        wandb_log,
        logdir,
    ) = load(config)

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, optimizer, logdir, len(input_size), min_is_best=True
    )

    for e in range(epoch, config["nepochs"] + epoch):
        last = False
        # Train 1 epoch
        (
            train_loss,
            gradient_norm,
            train_recon_loss,
            train_kld,
            mu_train,
            sigma_train,
            delta_train,
        ) = utils.train_epoch(
            model=model,
            loader=train_loader,
            f_loss=loss,
            optim=optimizer,
            device=device,
            config=config,
        )

        # Test
        test_loss, test_recon_loss, test_kld, mu_test, sigma_test, delta_test = (
            utils.test_epoch(
                model=model,
                loader=valid_loader,
                f_loss=loss,
                device=device,
                config=config,
            )
        )

        updated = model_checkpoint.update(
            epoch=e, score=test_loss, seed=seed, wandb_id=wandb.run.id, logdir=logdir
        )

        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"] + epoch,
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_MSE_loss": train_recon_loss,
            "test_MSE_loss": test_recon_loss,
            "train_KLD": train_kld,
            "test_KLD": test_kld,
            "mu_train": mu_train,
            "sigma_train": sigma_train,
            "delta_train": delta_train,
            "mu_test": mu_test,
            "sigma_test": sigma_test,
            "delta_test": delta_test,
            "gradient_norm": gradient_norm,
            "epoch": e,
        }

        image_path_valid = visualize_images(
            valid_loader, model, device, logdir, e, last, train=False
        )
        image_path_train = visualize_images(
            train_loader, model, device, logdir, e, last, train=True
        )

        imgs_valid = Image.open(image_path_valid)
        imgs_train = Image.open(image_path_train)
        # Log to wandb
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb.log(
                {
                    "generated_valid_images": [
                        wandb.Image(imgs_valid, caption="Epoch: {}".format(e))
                    ]
                }
            )
            wandb.log(
                {
                    "generated_train_images": [
                        wandb.Image(imgs_train, caption="Epoch: {}".format(e))
                    ]
                }
            )
            wandb.log(metrics)

    wandb.finish()


def visualize_images(data_loader, model, device, logdir, e, last=False, train=False):
    # Sample 5 images and their generated counterparts
    img_datasets = []
    img_gens = []

    for i, data in zip(range(5), iter(data_loader)):
        if isinstance(data, tuple) or isinstance(data, list):
            inputs, labels = data
        else:
            inputs = data
        img_dataset = inputs[random.randint(0, len(inputs) - 1)]
        if isinstance(model, VAE):
            img_gen = (
                model(img_dataset.unsqueeze_(0).to(device))[0].cpu().detach().numpy()
            )
        else:
            img_gen = model(img_dataset.unsqueeze_(0).to(device)).cpu().detach().numpy()
        img_datasets.append(img_dataset[0, :, :, :].numpy())
        img_gens.append(img_gen[0, :, :, :])
    if train:
        image_path = logdir / f"output_{e}_train.png"
    else:
        image_path = logdir / f"output_{e}_valid.png"
    # Call the modified show_image function
    if e % 10 == 0:
        last = True
    dt.show_images(img_datasets, img_gens, image_path, last)
    return image_path


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
