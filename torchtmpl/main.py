# coding: utf-8

# Standard imports
import logging
import sys
from os import path, makedirs, environ
import pathlib
import random

# External imports
import yaml
import wandb
import torch
import torch.nn as nn
import torchinfo.torchinfo as torchinfo
import torchcvnn.nn.modules as c_nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from PIL import Image
import numpy as np

# Local imports
from . import data as dt
from . import models
from . import optim
from . import utils
import torchtmpl as tl
from torchtmpl.models import VAE, UNet, AutoEncoder


def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    environ['MASTER_ADDR'] = 'localhost'
    environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


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


def train(config, rank, world_size):
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
    """
    data.delete_folders_with_few_pngs()
    print("Done")
    input()
    """
    # setup the process groups
    setup(rank, world_size)

    # seed_everything(2000)
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

    train_loader, valid_loader = dt.get_dataloaders(data_config, use_cuda, rank, world_size)


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

    '''
    # for parallelizing the model
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    '''
    # instantiate the model(it's your own model) and move it to the right device
    model = model.to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)


    # Build the loss
    logging.info("= Loss")
    loss = tl.optim.get_loss(config["loss"]["name"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = tl.optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["model"]["class"]

    if not path.isdir(logging_config["logdir"]):
        makedirs(logging_config["logdir"])

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

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, logdir, len(input_size), min_is_best=True
    )

    for e in range(config["nepochs"] + 1):
        torch.cuda.set_device(rank)
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
            loader=train_loader.sampler.set_epoch(e),
            f_loss=loss,
            optim=optimizer,
            device=device,
            config=config,
        )

        group_gather_train_loss = [torch.zeros_like(train_loss) for _ in range(world_size)]
        dist.all_gather(group_gather_train_loss, train_loss)
        train_loss = torch.mean(torch.stack(group_gather_train_loss))
        group_gather_gradient_norm = [torch.zeros_like(gradient_norm) for _ in range(world_size)]
        dist.all_gather(group_gather_gradient_norm, gradient_norm)
        gradient_norm = torch.mean(torch.stack(group_gather_gradient_norm))
        group_gather_train_recon_loss = [torch.zeros_like(train_recon_loss) for _ in range(world_size)]
        dist.all_gather(group_gather_train_recon_loss, train_recon_loss)
        train_recon_loss = torch.mean(torch.stack(group_gather_train_recon_loss))
        group_gather_train_kld = [torch.zeros_like(train_kld) for _ in range(world_size)]
        dist.all_gather(group_gather_train_kld, train_kld)
        train_kld = torch.mean(torch.stack(group_gather_train_kld))
        group_gather_mu_train = [torch.zeros_like(mu_train) for _ in range(world_size)]
        dist.all_gather(group_gather_mu_train, mu_train)
        mu_train = torch.mean(torch.stack(group_gather_mu_train))
        group_gather_sigma_train = [torch.zeros_like(sigma_train) for _ in range(world_size)]
        dist.all_gather(group_gather_sigma_train, sigma_train)
        sigma_train = torch.mean(torch.stack(group_gather_sigma_train))
        group_gather_delta_train = [torch.zeros_like(delta_train) for _ in range(world_size)]
        dist.all_gather(group_gather_delta_train, delta_train)
        delta_train = torch.mean(torch.stack(group_gather_delta_train))
                  
        # Test
        test_loss, test_recon_loss, test_kld, mu_test, sigma_test, delta_test = (
            utils.test_epoch(
                model=model,
                loader=valid_loader.sampler.set_epoch(e),
                f_loss=loss,
                device=device,
                config=config,
            )
        )

        group_gather_test_loss = [torch.zeros_like(test_loss) for _ in range(world_size)]
        dist.all_gather(group_gather_test_loss, test_loss)
        test_loss = torch.mean(torch.stack(group_gather_test_loss))
        group_gather_test_recon_loss = [torch.zeros_like(test_recon_loss) for _ in range(world_size)]
        dist.all_gather(group_gather_test_recon_loss, test_recon_loss)
        test_recon_loss = torch.mean(torch.stack(group_gather_test_recon_loss))
        group_gather_test_kld = [torch.zeros_like(test_kld) for _ in range(world_size)]
        dist.all_gather(group_gather_test_kld, test_kld)
        test_kld = torch.mean(torch.stack(group_gather_test_kld))
        group_gather_mu_test = [torch.zeros_like(mu_test) for _ in range(world_size)]
        dist.all_gather(group_gather_mu_test, mu_test)
        mu_test = torch.mean(torch.stack(group_gather_mu_test))
        group_gather_sigma_test = [torch.zeros_like(sigma_test) for _ in range(world_size)]
        dist.all_gather(group_gather_sigma_test, sigma_test)
        sigma_test = torch.mean(torch.stack(group_gather_sigma_test))
        group_gather_delta_test = [torch.zeros_like(delta_test) for _ in range(world_size)]
        dist.all_gather(group_gather_delta_test, delta_test)
        delta_test = torch.mean(torch.stack(group_gather_delta_test))
        
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

        # Sample 5 images and their generated counterparts
        img_datasets = []
        img_gens = []

        for i, data in zip(range(5), iter(valid_loader)):
            if isinstance(data, tuple) or isinstance(data, list):
                inputs, labels = data
            else:
                inputs = data
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
        if e % 10 == 0:
            last = True
        dt.show_images(img_datasets, img_gens, image_path, last)
        imgs = Image.open(image_path)

        # Log to wandb
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb.log(
                {"generated_images": [wandb.Image(imgs, caption="Epoch: {}".format(e))]}
            )
            wandb.log(metrics)

    wandb.finish()
    cleanup()


def test(config, rank, world_size):
    pass

def run(rank, world_size, config, func):
    # Wrapper function for multiprocessing
    if func == "train":
        train(config, rank, world_size)
    elif func == "test":
        test(config, rank, world_size)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]

    # Assume we have 3 GPUs
    world_size = config["world_size"]
    mp.spawn(
        run,
        args=(world_size, config, command,),
        nprocs=world_size,
        join=True
    )
