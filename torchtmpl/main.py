# coding: utf-8

# Standard imports
import logging
import sys
from os import path, makedirs, path
import pathlib
import shutil
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
import tqdm as tqdm

# Local imports
from . import data as dt
from . import models
from . import optim
from . import utils
import torchtmpl as tl
from torchtmpl.models import VAE, UNet, AutoEncoder, AutoEncoderWD


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

    log_path = config["logging"]["logdir"]

    if config["pretrained"]:
        seed = config["seed"]
        seed_everything(seed)
    else:
        seed = math.floor(random.random() * 10000)
        config["seed"] = seed
        seed_everything(seed)

    # dt.delete_folders_with_few_pngs(log_path=log_path)

    cdtype = torch.complex64
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        if config["pretrained"]:
            wandb.init(
                project=wandb_config["project"],
                resume="must",
                id=config["logging"]["wandb"]["run_id"],
                config=config,
            )
        else:
            wandb.init(project=wandb_config["project"], config=config)
            config["logging"]["wandb"]["run_id"] = wandb.run.id
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader = dt.get_dataloaders(data_config, use_cuda)

    # Load the checkpoint if needed
    if config["pretrained"]:
        checkpoint_path = log_path + "/last_model.pt"
        checkpoint = torch.load(checkpoint_path)
        logging.info(f"Loading checkpoint from {checkpoint_path}")

    # Build the model
    logging.info("= Model")

    model = models.build_model(config)
    model.apply(init_weights)

    if config["pretrained"]:
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

    if config["pretrained"]:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"] + 1  # Start from the next epoch

    # Copy the config file into the logdir
    # Let us use as base logname the class name of the model when wandb is not used
    logname = config["model"]["class"]
    if not path.isdir(log_path):
        makedirs(log_path)
    if config["pretrained"]:
        logdir = log_path
    else:
        if "wandb" in config["logging"]:
            logdir = log_path + "/" + logname + "_" + wandb.run.name
        else:
            logdir = utils.generate_unique_logpath(log_path, logname)
        if not path.isdir(logdir):
            makedirs(logdir)
        config["logging"]["logdir"] = logdir

    logging.info(f"Will be logging into {logdir}")

    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    if len(next(iter(train_loader))) == 2:
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
        + f"Train : {train_loader.dataset}\n"
        + f"Validation : {valid_loader.dataset}"
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
        wandb_log,
        logdir,
    )


def visualize_images(data_loader, model, device, logdir, e, last=False, train=False):

    model.eval()
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


def train(config):

    (
        model,
        optimizer,
        loss,
        train_loader,
        valid_loader,
        device,
        input_size,
        epoch,
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

        updated = model_checkpoint.update(epoch=e, score=test_loss)

        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"] + epoch,
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )
        model.eval()

        torch.save(
            {
                "epoch": e,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            },
            path.join(logdir, "last_model.pt"),
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


def test(config):

    log_path = config["logging"]["logdir"]

    cdtype = torch.complex64
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]
    data_config["batch_size"] = 1

    data_loader = dt.get_full_image_dataloader(data_config, use_cuda)

    # Load the checkpoint if needed
    if config["pretrained"]:
        checkpoint_path = log_path + "/best_model.pt"
        checkpoint = torch.load(checkpoint_path)
        logging.info(f"Loading checkpoint from {checkpoint_path}")

    # Build the model
    logging.info("= Model")

    model = models.build_model(config)
    model.apply(init_weights)

    if config["pretrained"]:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("No pretrained model available")

    model.to(device)

    if config["pretrained"]:
        logdir = log_path
    else:
        raise ValueError("No pretrained model available")

    logging.info(f"Will be logging into {logdir}")

    logdir = pathlib.Path(logdir)

    orginal_tensors = []
    for data in tqdm.tqdm(data_loader):
        if isinstance(data, tuple) or isinstance(data, list):
            inputs, labels = data
        else:
            inputs = data
        orginal_tensors.append(data.cpu().detach().numpy())

    original_image = dt.reassemble_image(
        segments=orginal_tensors,
        nb_cols=config["data"]["crop"]["end_col"] - config["data"]["crop"]["start_col"],
        nb_rows=config["data"]["crop"]["end_row"] - config["data"]["crop"]["start_row"],
        num_channels=config["data"]["num_channels"],
        segment_size=config["data"]["img_size"],
    )

    # Test
    reconstructed_tensors = utils.one_forward(
        model=model,
        loader=data_loader,
        device=device,
    )

    reconstructed_image = dt.reassemble_image(
        segments=reconstructed_tensors,
        nb_cols=config["data"]["crop"]["end_col"] - config["data"]["crop"]["start_col"],
        nb_rows=config["data"]["crop"]["end_row"] - config["data"]["crop"]["start_row"],
        num_channels=config["data"]["num_channels"],
        segment_size=config["data"]["img_size"],
    )

    dt.show_images(
        samples=original_image,
        generated=reconstructed_image,
        image_path=logdir / f"full_images.png",
        last=False,
    )


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if sys.argv[2] not in ["train", "retrain", "test"]:
        if sys.argv[2] == "train":
            if len(sys.argv) != 3:
                logging.error(f"Usage : {sys.argv[0]} config.yaml train")
                sys.exit(-1)
        else:
            if len(sys.argv) != 5:
                logging.error(
                    f"Usage : {sys.argv[0]} config.yaml retrain|test path_to_run"
                )
                sys.exit(-1)

    command = sys.argv[2]
    logging.info("Loading {}".format(sys.argv[1]))

    if command == "train":
        config = yaml.safe_load(open(sys.argv[1], "r"))
        config["pretrained"] = False
    else:
        path_to_run = sys.argv[3]
        config = yaml.safe_load(open(path_to_run + "/config.yml", "r"))
        config["pretrained"] = True
        if command == "retrain":
            command = "train"

    eval(f"{command}(config)")
