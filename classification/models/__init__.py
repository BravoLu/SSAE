from .generator import *
from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *

__factory = {
    'efficientnet': EfficientNetB0,
    'densenet': DenseNet121,
    'resnet': ResNet18,
    'vgg': VGG,
    'googlenet': GoogLeNet,
    'mobilenet': MobileNetV2,
}

def init_model(name, pre_dir, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))

    net = __factory[name](*args, **kwargs)
    checkpoint = torch.load(pre_dir)
    state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint
    change = False
    for k, v in state_dict.items():
        if k[:6] == 'module':
            change = True
            break
    if not change:
        new_state_dict = state_dict
    else:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    net.volatile = True
    return net

