from .baseline import Baseline 
from .generator import *
from .mgn import MGN 
from .IDE import * 
from .AlignedReID import *
from .PCB import * 
from .HACNN import *
from .DenseNet import *
from .MuDeep import * 
from .GD import MS_Discriminator 
from .fcn16s import *

__factory = {
    # 1 
    'ide': IDE, 
    'densenet121': DenseNet121,
    'aligned': ResNet50,
    'pcb': PCB,
    'mudeep': MuDeep,
    'hacnn': HACNN,

    'cam': IDE,
    'hhl': IDE, 
    'lsro': DenseNet121, 
    'spgan': IDE,

    'mgn': MGN, 
    'sbl': Baseline, 
}

def init_model(name, pre_dir, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))

    net = __factory[name](*args, **kwargs)
    checkpoint = torch.load(pre_dir, encoding='latin1')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint 
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
