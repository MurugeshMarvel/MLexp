from itertools import repeat
import collections.abc
try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def _float_to_int(x: float) -> int:
    """
    Symbolic tracing helper to substitute for inbuilt `int`.
    Hint: Inbuilt `int` can't accept an argument of type `Proxy`
    """
    return int(x)