import os
import torch
from train_helper_VMBNet import Trainer
# from config import parse_args
import numpy as np
import random


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 0:  # reproducible but slow
        torch.backends.cudnn.benchmark = False  # false by default, slow
        torch.backends.cudnn.deterministic = True  # Whether to use deterministic convolution algorithm? false by default.
    else:  # fast
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # setup_seed(43)
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
