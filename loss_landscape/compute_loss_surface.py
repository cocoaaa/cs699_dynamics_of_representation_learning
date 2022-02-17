"""
Here we compute loss at various points along the directions
We compute L(w+x*d1 + y*d2) at different values of x and y and save it to surface file which will be used to plot contours
"""
import argparse
import logging
import os
import sys

import dill
import numpy
import torch
from tqdm import tqdm

from train import get_dataloader
from utils.evaluations import get_loss_value
from utils.nn_manipulation import count_params, flatten_params, set_weights_by_direction
from utils.reproducibility import set_seed
from utils.resnet import get_resnet
from utils.fcnet import get_act_fn, get_fcnet

IN_SHAPE = (3, 32, 32) #nc, h,w of an input image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--seed", required=False, type=int, default=0)
    parser.add_argument("--gpu_id", required=True, type=int)

    parser.add_argument("--result_folder", "-r", required=True)
    parser.add_argument("--statefile", "-s", required=False, default=None)
    parser.add_argument(
        "--model", required=True, choices=["resnet20", "resnet32", "resnet44", "resnet56", \
                                          "fcnet3", "fcnet5", "fcnet10"]
    ) 
    parser.add_argument("--remove_skip_connections", action="store_true", default=False)
    parser.add_argument("--skip_bn_bias", action="store_true")

    # model parameters for fully-connected network
    parser.add_argument("--n_hidden", required=False, type=int)
    parser.add_argument(
        "--act", required=False, choices=["relu", "leaky", "softplus"], default='relu'
    )
    parser.add_argument("--use_bn", action="store_true", default=False)

    parser.add_argument("--batch_size", required=False, type=int, default=1028) #1000
    parser.add_argument("--direction_file", required=True, type=str)
    parser.add_argument(
        "--surface_file", type=str, required=True, help="filename to store evaluation results"
    )
    parser.add_argument(
        "--xcoords", type=str, default="51:-1:1",
        help="range of x-coordinate, specify as num:min:max"
    )
    parser.add_argument(
        "--ycoords", type=str, default="51:-1:1",
        help="range of y-coordinate, specify as num:min:max"
    )

    args = parser.parse_args()

    # set visible device
    # Select Visible GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", args.device, ", ", args.gpu_id)
    
    
    # set up logging
    os.makedirs(f"{args.result_folder}", exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    set_seed(args.seed)

    if os.path.exists(f"{args.result_folder}/{args.surface_file}"):
        logger.error(f"{args.surface_file} exists, so we will exit")
        sys.exit()

    # get dataset
    # using training dataset and only 10000 examples for faster evaluation
    train_loader, test_loader = get_dataloader(
        args.batch_size, transform_train_data=False, train_size=10000
    )

    # get model
    if "resnet" in args.model:
        model = get_resnet(args.model)(
            num_classes=10, remove_skip_connections=args.remove_skip_connections
        )
    elif "fcnet" in args.model:
        model_string = args.model
        in_feats = int(numpy.prod(IN_SHAPE))
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
    model.to(args.device)
    total_params = count_params(model)
    logger.info(f"using {args.model} with {total_params} parameters")

    logger.info(f"Loading model from {args.statefile}")
    state_dict = torch.load(args.statefile, pickle_module=dill, map_location=args.device)
    model.load_state_dict(state_dict)

    total_params = count_params(model, skip_bn_bias=args.skip_bn_bias)
    pretrained_weights = flatten_params(
        model, num_params=total_params, skip_bn_bias=args.skip_bn_bias
    ).to(args.device)

    logger.info(f"Loading directions from {args.direction_file}")
    temp = numpy.load(args.direction_file)
    direction1 = torch.tensor(temp["direction1"], device=args.device).float()
    direction2 = torch.tensor(temp["direction2"], device=args.device).float()

    x_num, x_min, x_max = [float(i) for i in args.xcoords.split(":")]
    y_num, y_min, y_max = [float(i) for i in args.ycoords.split(":")]

    x_num, y_num = int(x_num), int(y_num)

    logger.info(f"x-range: {x_min}:{x_max}:{x_num}")
    logger.info(f"y-range: {y_min}:{y_max}:{y_num}")

    xcoordinates = numpy.linspace(x_min, x_max, num=x_num)
    ycoordinates = numpy.linspace(y_min, y_max, num=y_num)

    losses = numpy.zeros((x_num, y_num))
    accuracies = numpy.zeros((x_num, y_num))

    with tqdm(total=x_num * y_num) as pbar:
        for idx_x, x in enumerate(xcoordinates):
            for idx_y, y in enumerate(ycoordinates):
                # import ipdb;ipdb.set_trace()
                set_weights_by_direction(
                    model, x, y, direction1, direction2, pretrained_weights,
                    skip_bn_bias=args.skip_bn_bias
                )
                losses[idx_x, idx_y], accuracies[idx_x, idx_y] = get_loss_value(
                    model, train_loader, args.device
                )
                pbar.set_description(f"x:{x: .4f}, y:{y: .4f}, loss:{losses[idx_x, idx_y]:.4f}")
                pbar.update(1)

    # save losses and accuracies evaluations
    logger.info("Saving results")
    numpy.savez(
        f"{args.result_folder}/{args.surface_file}", losses=losses, accuracies=accuracies,
        xcoordinates=xcoordinates, ycoordinates=ycoordinates
    )
