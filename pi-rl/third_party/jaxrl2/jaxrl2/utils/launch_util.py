# adapted from jaxrl2
from jaxrl2.utils.general_utils import AttrDict

def parse_training_args(train_args_dict, parser):
    for k, v in train_args_dict.items():
        if type(v) == tuple:
            parser.add_argument('--' + k, nargs="+", default=v, type=type(v[0]))
        elif type(v) != bool:
            parser.add_argument('--' + k, default=v, type=type(v))
        else:
            parser.add_argument('--' + k, default=int(v), type=int)
    args = parser.parse_args()
    config = {}
    for key in train_args_dict.keys():
        config[key] = getattr(args, key)
    variant = AttrDict(vars(args))
    variant['train_kwargs'] = config
    return variant, args
