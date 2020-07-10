import os
import random

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target+(1e-06)))

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        return Variable(tensor)

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1 and hasattr(layer, 'weight'):
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1 and hasattr(layer, 'weight'):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore):

    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        device = torch.device('cuda')
        net.to(device)
        net.restored = False

    # restore model weights
    if restore is not None and os.path.exists(restore):
        #net.load_state_dict(torch.load(restore))
        net.load_state_dict(torch.load(restore, map_location="cuda"))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    return net


def save_model(net, filename,params):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    # torch.save(net.state_dict(),
    #            os.path.join(params.model_root, filename))
    torch.save(net.module.state_dict(), os.path.join(params.model_root, filename))

    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))