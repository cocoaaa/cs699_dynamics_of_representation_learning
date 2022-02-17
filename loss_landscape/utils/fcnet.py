from argparse import ArgumentParser
import torch
import torch.nn as nn
import numpy as np
from typing import List, Iterable, Callable, Union, Optional, Any, TypeVar, Tuple, Dict


def make_fc_block(
         in_feats: int,
         out_feats: int,
         act_fn: Optional[Callable]=None,
         use_bn: Optional[bool]=False,
) -> nn.Module:
    act_fn = act_fn or nn.LeakyReLU(0.2, inplace=True)
    return nn.Sequential(
        nn.Linear(in_feats, out_feats),
        nn.BatchNorm1d(out_feats) if use_bn else nn.Identity(),
        act_fn
    )

class FCNet(nn.Module):

    def __init__(self,
#                  in_shape: Tuple[int, int, int],
                 in_feats: int,
                 n_hiddens: Iterable[int],
                 n_classes: Optional[int]=10,
                 act_fn: Optional[Callable]=None,
                 use_bn: Optional[bool]=False,
                 size_average: bool = False,
                 ):
        super().__init__()
        self.n_classes = n_classes
        self.act_fn = act_fn or nn.LeakyReLU(0.2, inplace=True)
        self.act_fn_name = str(act_fn)[3:] # removes nn module's prefix (e.g. `nn.<func-name>`)
        self.use_bn = use_bn

#         in_feats = int(np.prod(in_shape))
        self.n_feats = [in_feats, *n_hiddens]
        layers = [make_fc_block(in_, out_, self.act_fn, self.use_bn) \
                  for (in_, out_) in zip(self.n_feats, self.n_feats[1:])]

        self.model = nn.Sequential(
            *layers,
            nn.Linear(self.n_feats[-1], self.n_classes) #outputs logit (ie. scores for each class)
        )                                            # we output logit and use nn.BCEwithLogitsLoss
        self.size_average = size_average
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean', size_average=size_average)

        #     nn.Linear(int(np.prod(in_shape)), 512),
        #     self.act_fn,
        #     nn.Linear(512, 256),
        #     self.act_fn,
        #     nn.Linear(256, 1),  # prob(input belongs to class1), aka. "Logit"
        #     #             nn.Sigmoid(), #instead, we output logit and use nn.BCEwithLogitsLoss
        # )

    @property
    def name(self) -> str:
        bn = f'FC_{self.n_layers}-act_{self.act_fn_name}-bn_{self.use_bn}'
        return bn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the score for being class 1
        ie. logit, not a probability
        """
        img_flat = x.view(x.size(0), -1)
        return self.model(img_flat)

    def compute_loss(self, score: torch.Tensor, target: torch.Tensor):
        return self.loss_fn(score, target)

    
def basic_fcnet(in_feats: int, n_layers: int, n_hidden: int, act_fn: Callable, use_bn: bool):
    n_hiddens = [n_hidden] * n_layers
    
    return FCNet(in_feats=in_feats, n_hiddens=n_hiddens, act_fn=act_fn, use_bn=use_bn)

    
def fcnet3(in_feats: int, n_hidden: int, act_fn: Callable, use_bn: bool):
    n_layers = 2
    return basic_fcnet(in_feats, n_layers, n_hidden, act_fn, use_bn)


def fcnet5(in_feats: int, n_hidden: int, act_fn: Callable, use_bn: bool):
    n_layers = 4
    return basic_fcnet(in_feats, n_layers, n_hidden, act_fn, use_bn)

def fcnet10(in_feats: int, n_hidden: int, act_fn: Callable, use_bn: bool):
    n_layers = 9
    return basic_fcnet(in_feats, n_layers, n_hidden, act_fn, use_bn)


def get_fcnet(model_string, **kwargs):
    if model_string == "fcnet3":
        return fcnet3(**kwargs)

    if model_string == "fcnet5":
        return fcnet5(**kwargs)

    if model_string == "fcnet10":
        return fcnet10(**kwargs)

    
def get_act_fn(act_name: str):
    act_name = act_name.lower()
    
    if act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'leaky':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act_name == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError("act function must be one of relu, leaky, softplus")

if __name__ == '__main__':

    image = torch.rand(4, 3, 32, 32)
    
    # model
#     in_feats = int(np.prod(image.shape[1:]))
#     n_hidden = 32
# #     act_fn = nn.LeakyReLU(0.1, inplace=True)
# #     act_fn = nn.ReLU()
#     act_fn = nn.Softplus()
#     use_bn = True
    
#     model = fcnet3(in_feats, n_hidden=n_hidden, act_fn=act_fn, use_bn=use_bn)    



    # alternatively
    parser = ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=["fcnet3", "fcnet5", "fcnet10"]
    )
    parser.add_argument("--n_hidden", required=True, type=int)

    parser.add_argument(
        "--act", required=True, choices=["relu", "leaky", "softplus"]
    )
    parser.add_argument("--use_bn", action="store_true", default=False)

    parser.add_argument("--batch_size", required=False, type=int, default=128)

    

    args = parser.parse_args()
    model_string = args.model
    in_feats = int(np.prod(image.shape[1:]))
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
    print('use bn: ', use_bn)
    print(model)

    
    # forward
    output = model(image)
    print(output.shape)

    
# Test
# python fcnet.py --model fcnet3 --n_hidden 16 --act relu  
#python fcnet.py --model fcnet3 --n_hidden 16 --act relu  --use_bn