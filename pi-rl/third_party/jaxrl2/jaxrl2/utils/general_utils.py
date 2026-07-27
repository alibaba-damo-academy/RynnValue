# adapted from jaxrl2
class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self): return self
    def __setstate__(self, d): self = d

def add_batch_dim(input):
    if isinstance(input, dict):
        for k, v in input.items():
            input[k] = v[None]
    else:
        input = input[None]
    return input
