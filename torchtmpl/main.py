# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import random

# External imports
import yaml
import wandb
import torch
import torch.nn as nn
import torchinfo.torchinfo as torchinfo
import torchcvnn.nn.modules as c_nn
from PIL import Image

# Local imports
from . import data
from . import models
from . import optim
from . import utils
import torchtmpl as tl
from torchtmpl.models import VAE, UNet, AutoEncoder


def init_weights(m):
    if isinstance(m, nn.Linear):
        c_nn.init.complex_kaiming_normal_(m.weight, nonlinearity="relu")
        # nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


def train(config):
    """
    Train function

    Sample output :
        ```.bash
        (venv) me@host:~$ python mnist.py
        Logging to ./logs/CMNIST_0
        >> Training
        100%|██████| 844/844 [00:17<00:00, 48.61it/s]
        >> Testing
        [Step 0] Train : CE  0.20 Acc  0.94 | Valid : CE  0.08 Acc  0.97 | Test : CE 0.06 Acc  0.98[>> BETTER <<]

        >> Training
        100%|██████| 844/844 [00:16<00:00, 51.69it/s]
        >> Testing
        [Step 1] Train : CE  0.06 Acc  0.98 | Valid : CE  0.06 Acc  0.98 | Test : CE 0.05 Acc  0.98[>> BETTER <<]

        >> Training
        100%|██████| 844/844 [00:15<00:00, 53.47it/s]
        >> Testing
        [Step 2] Train : CE  0.04 Acc  0.99 | Valid : CE  0.04 Acc  0.99 | Test : CE 0.04 Acc  0.99[>> BETTER <<]

        [...]
        ```

    """
    cdtype = torch.complex64
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_size = tl.data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config
    model = models.build_model(model_config)
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
    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.numel())
    input()
    """

    # Build the loss
    logging.info("= Loss")
    loss = tl.optim.get_loss(config["loss"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = tl.optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["model"]["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open("config.yml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
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

    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, logdir, len(input_size), min_is_best=True
    )

    last = False
    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train_epoch(
            model=model,
            loader=train_loader,
            f_loss=loss,
            optim=optimizer,
            device=device,
        )

        # Test
        test_loss = utils.test_epoch(
            model=model,
            loader=valid_loader,
            f_loss=loss,
            device=device,
        )

        updated = model_checkpoint.update(test_loss)

        logging.info(
            "[%d/%d] Test loss : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_loss,
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {"train_MSE": train_loss, "test_MSE": test_loss}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)

        # Sample 5 images and their generated counterparts
        img_datasets = []
        img_gens = []

        for i, inputs in zip(range(5), iter(valid_loader)):
            img_dataset = inputs[random.randint(0, len(inputs) - 1)]
            if isinstance(model, VAE):
                img_gen = (
                    model(img_dataset.unsqueeze_(0).to(device))[0]
                    .cpu()
                    .detach()
                    .numpy()
                )
                # img_gens.append(img_gen[0, :, :, :])
            else:
                img_gen = (
                    model(img_dataset.unsqueeze_(0).to(device)).cpu().detach().numpy()
                )
                # img_gens.append(img_gen)
            img_datasets.append(img_dataset[0, :, :, :].numpy())
            img_gens.append(img_gen[0, :, :, :])

        image_path = logdir / f"output_{e}.png"
        # Call the modified show_image function
        if e == config["nepochs"] - 1:
            last = True
        data.show_images(img_datasets, img_gens, image_path, last)
        imgs = Image.open(image_path)
        # Log the image to wandb
        if wandb_log is not None:
            logging.info("Logging image on wandb")
            wandb.log(
                {"generated_images": [wandb.Image(imgs, caption="Epoch: {}".format(e))]}
            )

    wandb.finish()


"""
def test_imgs():
    # Dataloading
    train_valid_dataset = torchcvnn.datasets.dataPolSFAlos2Dataset()
    # Train dataloader
    train_dataset = torch.utils.data.Subset(
        train_valid_dataset, list(range(len(train_valid_dataset)))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = AutoEncoder(num_channels=3)
    autoencoder.load_state_dict(torch.load("./logs/AutoEncoder_0/best_model.pt"))
    # autoencoder = onnx.load("my_image_classifier.onnx")
    # onnx.checker.check_model(autoencoder)
    autoencoder.to(device)

    for i in range(0, 10):
        i = 1
        stack, _ = train_dataset[i]
        show_image(stack.numpy())
        show_image(
            autoencoder(stack[None, :, :, :].to(device))
            .cpu()
            .detach()
            .numpy()[0, :, :, :]
        )
"""

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
