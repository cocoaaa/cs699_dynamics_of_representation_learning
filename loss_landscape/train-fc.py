"""
Script to train a neural network:
    currently supports training resnet for CIFAR-10 with and w/o skip connections

    Also does additional things that we may need for visualizing loss landscapes, such as using
      frequent directions or storing models during the executions etc.
   This has limited functionality or options, e.g., you do not have options to switch datasets
     or architecture too much.
"""

import argparse
import logging
import os
import pprint
import time
from pathlib import Path

import dill
import numpy as np
import numpy.random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
# ipython bug
from utils.evaluations import get_loss_value
from utils.linear_algebra import FrequentDirectionAccountant
from utils.nn_manipulation import count_params, flatten_grads
from utils.reproducibility import set_seed
from utils.fcnet import get_act_fn, get_fcnet
from utils.misc import now2str

# "Fixed" hyperparameters
NUM_EPOCHS = 200
# In the resnet paper they train for ~90 epoch before reducing LR, then 45 and 45 epochs.
# We use 100-50-50 schedule here.
# LR = 0.01
DATA_FOLDER = "../data/"
IN_SHAPE = (3, 32, 32) #nc, h,w of an input image

def get_dataloader(batch_size, train_size=None, test_size=None, transform_train_data=True):
    """
        returns: cifar dataloader

    Arguments:
        batch_size:
        train_size: How many samples to use of train dataset?
        test_size: How many samples to use from test dataset?
        transform_train_data: If we should transform (random crop/flip etc) or not
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
            transforms.ToTensor(), normalize
        ]
    ) if transform_train_data else transforms.Compose([transforms.ToTensor(), normalize])

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_FOLDER, train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_FOLDER, train=False, transform=test_transform, download=True
    )

    if train_size:
        indices = numpy.random.permutation(numpy.arange(len(train_dataset)))
        train_dataset = Subset(train_dataset, indices[:train_size])

    if test_size:
        indices = numpy.random.permutation(numpy.arange(len(test_dataset)))
        test_dataset = Subset(train_dataset, indices[:test_size])

    # Data loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)

    parser.add_argument("--gpu_id", required=True, type=int)
#     parser.add_argument(
#         "--device", required=False, default="cuda" if torch.cuda.is_available() else "cpu"
#     )
    parser.add_argument("--result_folder", "-r", required=True)
    parser.add_argument(
        "--mode", required=False, nargs="+", choices=["test", "train"], default=["test", "train"]
    )

    # model related arguments
    parser.add_argument("--statefile", "-s", required=False, default=None)
    parser.add_argument(
        "--model", required=True, choices=["fcnet3", "fcnet5", "fcnet10"]
    )
    # parameters for fully-connected network
    parser.add_argument("--n_hidden", required=True, type=int)
    parser.add_argument(
        "--act", required=True, choices=["relu", "leaky", "softplus"]
    )
    parser.add_argument("--use_bn", action="store_true", default=False)
    parser.add_argument(
        "--skip_bn_bias", action="store_true",
        help="whether to skip considering bias and batch norm params or not, Li et al do not consider bias and batch norm params"
    )

    parser.add_argument("--batch_size", required=False, type=int, default=128)
    parser.add_argument("--lr", required=False, type=float, default=0.1)

    parser.add_argument(
        "--save_strategy", required=False, nargs="+", choices=["epoch", "init"],
        default=["epoch", "init"]
    )

    args = parser.parse_args()

    # set visible device
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", args.device, ", ", args.gpu_id)
    
    # set up logging
    uid = now2str()
    log_dir = Path(args.result_folder)/args.model/f'run_{uid}'
    ckpt_save_dir = log_dir / "ckpts"
    fds_save_dir = log_dir / "fds"

    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    fds_save_dir.mkdir(parents=True, exist_ok=True)
    
    
#     os.makedirs(f"{args.result_folder}/ckpt", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    summary_writer = SummaryWriter(log_dir=log_dir)
    logger.info("Config:")
    logger.info(pprint.pformat(vars(args), indent=4))

    set_seed(args.seed)

    # get dataset
    train_loader, test_loader = get_dataloader(args.batch_size)

    # get model
    model_string = args.model
    in_feats = int(np.prod(IN_SHAPE))
    n_hidden = args.n_hidden
    act_fn = get_act_fn(args.act)
    use_bn = args.use_bn
    model_args = {
        'in_feats':in_feats,
        'n_hidden': n_hidden,
        'act_fn':act_fn,
        'use_bn':use_bn
    }
    model = get_fcnet(model_string, **model_args)
#     print('use bn: ', use_bn)
#     print(model)
    model.to(args.device)
    logger.info(f"using {args.model} with {count_params(model)} parameters")

    logger.debug(model)

    # we can try computing principal directions from some specific training rounds only
    total_params = count_params(model, skip_bn_bias=args.skip_bn_bias)
    fd = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)
    # frequent direction for last 10 epoch
    fd_last_10 = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)
    # frequent direction for last 1 epoch
    fd_last_1 = FrequentDirectionAccountant(k=2, l=10, n=total_params, device=args.device)

    # use the same setup as He et al., 2015 (resnet)
    LR = 0.1 if 'resnet' in model_string else args.lr 
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lambda x: 1 if x < 100 else (0.1 if x < 150 else 0.01)
    )

    if "init" in args.save_strategy:
        torch.save(
            model.state_dict(), ckpt_save_dir/"init_model.pt", pickle_module=dill
        )

    # training loop
    # we pass flattened gradients to the FrequentDirectionAccountant before clearing the grad buffer
    total_step = len(train_loader) * NUM_EPOCHS
    step = 0
    direction_time = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Forward pass
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # get gradient and send it to the accountant
            start = time.time()
            fd.update(flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias))
            direction_time += time.time() - start

            if epoch >= NUM_EPOCHS - 10:
                fd_last_10.update(
                    flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias)
                )
            if epoch >= NUM_EPOCHS - 1:
                fd_last_1.update(
                    flatten_grads(model, total_params, skip_bn_bias=args.skip_bn_bias)
                )

            summary_writer.add_scalar("train/loss", loss.item(), step)
            step += 1

            if step % 100 == 0:
                logger.info(
                    f"Epoch [{epoch}/{NUM_EPOCHS}], Step [{step}/{total_step}] Loss: {loss.item():.4f}"
                )

        scheduler.step()

        # Save the model checkpoint
        if "epoch" in args.save_strategy:
            torch.save(
                model.state_dict(), ckpt_save_dir/f'{epoch + 1}_model.pt',
                pickle_module=dill
            )

        loss, acc = get_loss_value(model, test_loader, device=args.device)
        logger.info(f'Accuracy of the model on the test images: {100 * acc}%')
        summary_writer.add_scalar("test/acc", acc, step)
        summary_writer.add_scalar("test/loss", loss, step)

    logger.info(f"Time to computer frequent directions {direction_time} s")

    logger.info(f"fd was updated for {fd.step} steps")
    logger.info(f"fd_last_10 was updated for {fd_last_10.step} steps")
    logger.info(f"fd_last_1 was updated for {fd_last_1.step} steps")

    # save the frequent_direction buffers and principal directions
    buffer = fd.get_current_buffer()
    directions = fd.get_current_directions()
    directions = directions.cpu().data.numpy()

    numpy.savez(
        fds_save_dir/"buffer.npy",
        buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
    )

    # save the frequent_direction buffer
    buffer = fd_last_10.get_current_buffer()
    directions = fd_last_10.get_current_directions()
    directions = directions.cpu().data.numpy()

    numpy.savez(
        fds_save_dir/"buffer_last_10.npy",
        buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
    )

    # save the frequent_direction buffer
    buffer = fd_last_1.get_current_buffer()
    directions = fd_last_1.get_current_directions()
    directions = directions.cpu().data.numpy()

    numpy.savez(
        fds_save_dir/"buffer_last_1.npy",
        buffer=buffer.cpu().data.numpy(), direction1=directions[0], direction2=directions[1]
    )

    
# nohup python train-fc.py --gpu_id 2 --result_folder '../results/hw1-exp4-run1' --mode train  \
# --model fcnet3 --n_hidden 64 --act relu \
# --batch_size 128 --lr 1e-2 > ./training-logs/log-exp4-run1-fcnet_3-n_hidden_64-act_relu_bs_128.txt &