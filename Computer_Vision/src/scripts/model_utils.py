from typing import Callable, List, Dict, Union, Optional
from torch import nn
from functools import partial
import math
from copy import deepcopy

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def register_notrace_function(func: Callable):
    """
    Decorator for functions which ought not to be traced through
    """
    _autowrap_functions = set()
    _autowrap_functions.add(func)
    return func

def get_pretrained_cfg(model_name):
    if model_name in _model_pretrained_cfgs:
        return deepcopy(_model_pretrained_cfgs[model_name])
    return {}

def resolve_pretrained_cfg(variant: str, pretrained_cfg=None, kwargs=None):
    if pretrained_cfg and isinstance(pretrained_cfg, dict):
        # highest priority, pretrained_cfg available and passed explicitly
        return deepcopy(pretrained_cfg)
    if kwargs and 'pretrained_cfg' in kwargs:
        # next highest, pretrained_cfg in a kwargs dict, pop and return
        pretrained_cfg = kwargs.pop('pretrained_cfg', {})
        if pretrained_cfg:
            return deepcopy(pretrained_cfg)
    # lookup pretrained cfg in model registry by variant
    pretrained_cfg = get_pretrained_cfg(variant)
    assert pretrained_cfg
    return pretrained_cfg

# def build_model_with_cfg(
#         model_cls: Callable,
#         variant: str,
#         pretrained: bool,
#         pretrained_cfg: Optional[Dict] = None,
#         model_cfg: Optional[Any] = None,
#         feature_cfg: Optional[Dict] = None,
#         pretrained_strict: bool = False,
#         pretrained_filter_fn: Optional[Callable] = None,
#         pretrained_custom_load: bool = False,
#         kwargs_filter: Optional[Tuple[str]] = None,
#         **kwargs):
#     """ Build model with specified default_cfg and optional model_cfg
#     This helper fn aids in the construction of a model including:
#       * handling default_cfg and associated pretrained weight loading
#       * passing through optional model_cfg for models with config based arch spec
#       * features_only model adaptation
#       * pruning config / model adaptation
#     Args:
#         model_cls (nn.Module): model class
#         variant (str): model variant name
#         pretrained (bool): load pretrained weights
#         pretrained_cfg (dict): model's pretrained weight/task config
#         model_cfg (Optional[Dict]): model's architecture config
#         feature_cfg (Optional[Dict]: feature extraction adapter config
#         pretrained_strict (bool): load pretrained weights strictly
#         pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
#         pretrained_custom_load (bool): use custom load fn, to load numpy or other non PyTorch weights
#         kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
#         **kwargs: model args passed through to model __init__
#     """
#     pruned = kwargs.pop('pruned', False)
#     features = False
#     feature_cfg = feature_cfg or {}

#     # resolve and update model pretrained config and model kwargs
#     pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=pretrained_cfg)
#     update_pretrained_cfg_and_kwargs(pretrained_cfg, kwargs, kwargs_filter)
#     pretrained_cfg.setdefault('architecture', variant)

#     # Setup for feature extraction wrapper done at end of this fn
#     if kwargs.pop('features_only', False):
#         features = True
#         feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
#         if 'out_indices' in kwargs:
#             feature_cfg['out_indices'] = kwargs.pop('out_indices')

#     # Build the model
#     model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
#     model.pretrained_cfg = pretrained_cfg
#     model.default_cfg = model.pretrained_cfg  # alias for backwards compat
    
#     if pruned:
#         model = adapt_model_from_file(model, variant)

#     # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
#     num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
#     if pretrained:
#         if pretrained_custom_load:
#             # FIXME improve custom load trigger
#             load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
#         else:
#             load_pretrained(
#                 model,
#                 pretrained_cfg=pretrained_cfg,
#                 num_classes=num_classes_pretrained,
#                 in_chans=kwargs.get('in_chans', 3),
#                 filter_fn=pretrained_filter_fn,
#                 strict=pretrained_strict)

#     # Wrap the model in a feature extraction module if enabled
#     if features:
#         feature_cls = FeatureListNet
#         if 'feature_cls' in feature_cfg:
#             feature_cls = feature_cfg.pop('feature_cls')
#             if isinstance(feature_cls, str):
#                 feature_cls = feature_cls.lower()
#                 if 'hook' in feature_cls:
#                     feature_cls = FeatureHookNet
#                 elif feature_cls == 'fx':
#                     feature_cls = FeatureGraphNet
#                 else:
#                     assert False, f'Unknown feature class {feature_cls}'
#         model = feature_cls(model, **feature_cfg)
#         model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)  # add back default_cfg
#         model.default_cfg = model.pretrained_cfg  # alias for backwards compat
    
#     return model

def named_apply(fn: Callable, module: nn.Module, name='', depth_first=True, include_root=False) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = '.'.join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name,
                    depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module
    
def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.
    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.
    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x