"""liNeN: lit neural-nets"""
import sys
import os
import functools
from importlib import reload
from pprint import pprint
import inspect
from contextlib import contextmanager
import dataclasses

from typing import Any, Callable, Sequence, Iterable, List, Optional, Tuple, Type, Union, TypeVar

import jax
from jax import numpy as jnp, random, lax
from jax import tree_util
import numpy as np

import flax
from flax import nn
from flax.nn import initializers
from flax import traverse_util
from flax import serialization

from flax.core import Scope, init, apply, lift, Array
from flax.core.scope import _unfreeze_variables, Variable, _fold_in_str
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict

from .dotgetter import DotGetter

PRNGKey = Any
Array = Any
T = TypeVar('T')

# pylint: disable=protected-access,attribute-defined-outside-init

# Utilities for autonaming pytrees of Modules defined inside setup()
# -----------------------------------------------------------------------------
def is_module_tree(in_tree):
  """Determine if in_tree is a pytree of subclasses of Module."""
  # reject trivial pytrees, {}, [], (), etc.
  if not tree_util.tree_leaves(in_tree):
    return False
  reduce_fn = lambda prev, cur: prev and isinstance(cur, Module)
  return jax.tree_util.tree_reduce(reduce_fn, in_tree, True)

def get_suffix_module_pairs(module_tree):
  """Helper for naming pytrees of submodules."""
  if isinstance(module_tree, Module):
    return [('', module_tree)]
  else:
    flat_tree = traverse_util.flatten_dict(
        serialization.to_state_dict(module_tree))
    return [('_' + '_'.join(k), v) for k, v in flat_tree.items()]

# Gathers all names on object.
# -----------------------------------------------------------------------------
def all_names_on_object(x):
  """Get all names on self and on classes throughout MRO."""
  nameset = set(x.__dict__.keys())
  for cls in x.__class__.__mro__:
    nameset = nameset.union(set(cls.__dict__.keys()))
  return nameset

# Method wrapping of "compact methods" and setup()
# -----------------------------------------------------------------------------
def compact(fun):
  """Allow inline submodules and parameters within a single Module method."""
  @functools.wraps(fun)
  def wrapped_method(self, *args, **kwargs):
    # Store compact method *name* rather than reference, so that subclassing
    # a module with a @compact __init__ method is still possible
    if getattr(self, '_compact_method', fun.__name__) != fun.__name__:
      raise RuntimeError(
        'Only one method per class can be @compact. You can remove @compact and define '
        'submodules and variables in setup(), or use two separate modules.')
    object.__setattr__(self, '_compact_method', fun.__name__)

    if self.scope is None:
      raise ValueError("Can't call methods on orphaned modules")

    object.__setattr__(self, '_in_compact_method', True)
    try:
      return fun(self, *args, **kwargs)
    finally:
      object.__setattr__(self, '_in_compact_method', False)
      object.__setattr__(self, '_reservations', set())
      object.__setattr__(self, '_autoname_cursor', dict())
      object.__setattr__(self, '_last_varname', None)
      object.__setattr__(self, 'scope', self.scope.rewound())
  return wrapped_method

def wrap_setup(fun):
  @functools.wraps(fun)
  def wrapped_setup_method(self, *args, **kwargs):
    object.__setattr__(self, '_in_setup', True)
    try:
      return fun(self, *args, **kwargs)
    finally:
      object.__setattr__(self, '_in_setup', False)
  return wrapped_setup_method


# Base Module definition.
# -----------------------------------------------------------------------------
class Module:
  """Base Module Class"""
  # class defaults -- Note: these __must not__ be made annotations!
  _in_compact_method = False
  _in_call = False
  _in_setup = False
  _last_varname = None
  scope = None

  @classmethod
  def __init_subclass__(cls):
    """Automatically initialize all subclasses as custom dataclasses."""
    # All Flax Modules are dataclasses.  We force this convention since
    # it encourages the stateless behavior needed to clone module instances for
    # functional transformation.  Instead of using a python metaclass, we
    # automatically transform Modules into dataclasses at subclass creation
    # time, and we set the first dataclass argument to `parent`, and the last
    # optional argument to `name`.
    cls._add_parent_and_name_attrs()
    dataclasses.dataclass(cls)
    cls.setup = wrap_setup(cls.setup)

  @classmethod
  def _add_parent_and_name_attrs(cls):
    """Add dataclass attributes: `parent` first, required and `name` last, optional."""
    annotations = cls.__dict__.get('__annotations__', {})
    if 'parent' in annotations or 'name' in annotations:
      raise ValueError(
          f'properties `parent` and `name` are reserved: {annotations}')
    # Add `parent` and `name` default fields at beginning and end, resp.
    new_annotations = {}
    if 'parent' not in getattr(cls, '__dataclass_fields__', {}):
      new_annotations.update(
          {'parent': Union[Type["Module"], Type["Scope"], None]})
      cls.parent = dataclasses.field(repr=False)
    new_annotations.update(annotations)
    if 'name' in getattr(cls, '__dataclass_fields__', {}):
      cls.__dataclass_fields__.pop('name')
    new_annotations['name'] = str
    cls.__annotations__ = new_annotations
    cls.name = None  # default value of name is None.

  def __setattr__(self, name: str, val: Any):
    """We overload setattr solely to support pythonic naming via
    assignment of submodules in the special setup() function:
      self.submodule_name = MyModule(...)
    we also support lists and other general pytrees, e.g.:
      self.submodules = [MyModule0(..), MyModule1(..), ...]
    """
    # val is a Module or pytree whose leaves are all Modules.
    if is_module_tree(val):
      # We don't mess with the parent module.
      if name == 'parent':
        pass
      # Modules have been passed in as dataclass args.
      # if they're orphaned, i.e. Module(None, ...) then attach them.
      # TODO: should we not do this immediately, and give the user more
      #       control over when such "module defs" are attached?
      elif name in self.__dataclass_fields__.keys():
        for suffix, submodule in get_suffix_module_pairs(val):
          if submodule.parent is None:
            submodule.parent = self
            if submodule.name is not None:
              raise ValueError(
                  "In setup assign names via self.<name> assignment.")
            submodule.name = f'{name}{suffix}'
            submodule.__post_init__()
      # Submodules are being defined and attached in setup()
      else:
        if not self._in_setup:
          raise ValueError("You can only assign submodules to self in setup().")
        for suffix, submodule in get_suffix_module_pairs(val):
          if submodule.parent is None:
            submodule.parent = self
          elif submodule.parent != self:
            raise ValueError("Can't attach to remote parent in setup, pass in "
                             "bound Modules from outside as an argument.")
          if submodule.name is not None:
            raise ValueError(
                "In setup assign names via self.<name> assignment.")
          submodule.name = f'{name}{suffix}'
          submodule.__post_init__()
    # val is a parameter array or a Variable reference class.
    elif isinstance(val, (np.ndarray, jax.interpreters.xla.DeviceArray,
                          Variable)):
      if self._last_varname and self._last_varname != name:
        raise ValueError(f'Variable name {self._last_varname} must equal'
                         f' attribute name {name}.')
      self._last_varname = None
    # Finally, always run default __setattr__ to attach to self.__dict__.
    object.__setattr__(self, name, val)

  def __post_init__(self):
    # In dataclasses, __init__ is overridden to process dataclass arguments,
    # and __post_init__ is called immediately afterwards.  Here, depending
    # on the type of `parent` passed to initialize the Module, we either
    # defer initialization, attach this Module as a submodule of a parent,
    # or bind this Module at the top-level to variables and rngs.

    self._autoname_cursor = dict()  # autonaming state
    self._reservations = set()  # autonaming state
    self.children = dict()  # tracks child modules

    # Initialization is deferred until attachment by __setattr__ for orphan
    # Modules i.e. MyModule(None, ...)
    if self.parent is None:
      return

    # Register submodule on parent Module.
    if isinstance(self.parent, Module):
      # When initializing an unnamed Module inside setup()
      # initialization is deferred until attachment by __setattr__
      # i.e. self.mymodule = MyModule(self, ...)
      if self.parent._in_setup and self.name is None:
        return
      if not self.parent._initialization_allowed:
        raise ValueError(
          'Submodules must be defined in `setup()` or in a method wrapped in `@compact`')
      # Autonaming of submodules.
      if self.name is None:
        prefix = f"{self.__class__.__name__}"
        cursor = self.parent._autoname_cursor.get(prefix, 0)
        self.name = f"{prefix}_{cursor}"
        self.parent._autoname_cursor[prefix] = cursor + 1
      if self.parent._name_taken(self.name):
        raise ValueError(
            f"A variable of name {self.name} exists already, or "
            f"trying to share submodule {self.__class__.__name__} by name "
            f"{self.name}. To share submodules, store module instances as a"
            f" Python object or as an attribute on self and reuse.")
      self.parent._reservations.add(self.name)
      self.parent.children[self.name] = self
      self.scope = self.parent.scope.push(self.name)
      # TODO: find cleaner way to opt-out of core name collision check
      self.parent.scope.reservations.remove(self.name)

    # Top-level invocation with a functional Scope.
    elif isinstance(self.parent, Scope):
      self.scope = self.parent

    else:
      raise ValueError("parent must be None, Module or Scope")

    # Call the user-defined initialization function.
    self.setup()

  def setup(self):
    """Called when Module instance receives variables and PRNGs.

    If you want to use PRNGs and/or read or write variables during module
    instance construction, override this method. It will be automatically
    called from the dataclass `__post_init__`.
    """
    pass

  def _name_taken(self, name):
    return name in self._reservations or name in all_names_on_object(self)

  @property
  def _initialization_allowed(self):
    return self._in_setup or self._in_compact_method

  def clone(self, **updates):
    attrs = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
    attrs.update(**updates)
    return self.__class__(**attrs)

  # TODO: Should this be what `clone` always does if you don't pass in an explicit
  # parent?
  def detached(self):
    return self.clone(parent=None)

  # TODO: Consider whether this is a helpful abstraction, and think about naming.
  # See its use in design_test/linen/weight_std.py
  def materialized(self, variables={}, rngs={}):
    assert self.scope is None, ("Can't attach a module twice."
                                " Maybe you want to clone first?")
    return self.clone(parent=Scope(variables, rngs))

  def variable(self, kind: str, name: str, init_fn, *init_args):
    if not self._initialization_allowed:
      raise ValueError(
        'Variables must be initialized in `setup()` or in a method wrapped in `@compact`')
    if self._name_taken(name):
      raise ValueError(
          f'Name {name} already in use in {self.__class__.__name__}.')
    self._reservations.add(name)
    # ephemeral state for setattr name-equality-check
    self._last_varname = None if self._in_call else name
    v = self.scope.variable(kind, name, init_fn, *init_args)
    # TODO: find cleaner way to opt-out of core name collision check
    self.scope.reservations.remove(name)
    self.children[name] = kind
    return v

  def has_variable(self, kind: str, name: str):
    return self.scope.has_variable(kind, name)

  def param(self, name: str, init_fn: Callable[..., T], *init_args,
            kind='param') -> T:
    p_init_fn = lambda *args: init_fn(self.make_rng('param'), *args)
    return self.variable(kind, name, p_init_fn, *init_args).value

  def make_rng(self, kind: str) -> PRNGKey:
    return self.scope.make_rng(kind)

  @property
  def variables(self):
    """Get a view of Module variables with easy dot-syntax navigation."""
    return DotGetter(self.scope.variables())

  def __getattr__(self, name):
    # Used for easy colab/jupyter introspection, and to provide a
    # consistent top-level interface to self.<attr> for both simple
    # and multi-method modules.
    if name in self.children:
      val = self.children[name]
      if isinstance(val, str):  # variable
        return self.variables[val][name]
      else:  # submodule
        val.scope = self.scope.push(name)
        self.scope.reservations.remove(name)
        return val
    else:
      raise AttributeError(
          f"'{self.__class__.__name__}' object has no attribute '{name}'")

  def __dir__(self):
    return list(self.children.keys()) + object.__dir__(self)

  @contextmanager
  def mutate(self, mutable=True, **updates):
    cloned = self.clone(**updates)
    try:
      cloned.scope._variables = _unfreeze_variables(
          cloned.scope._variables, mutable)
      yield cloned
    finally:
      cloned.scope._variables = freeze(cloned.scope._variables)

  def initialized(self, rngs, *args, method='__call__', **kwargs):
    if self.parent is not None:
      raise ValueError("Pattern for initialized is "
                       "`Module(parent=None, ...attrs...).initialized(...)`")
    scope = Scope(variables={}, rngs=rngs)
    with self.mutate(parent=scope) as initialized:
      if method is not None:
        getattr(initialized, method)(*args, **kwargs)
    return initialized

  def vars_init(self, *args, **kwargs):
    return self.initialized(*args, **kwargs).variables

  def apply(self, variables, *args, rngs=None, method='__call__', mutable=False, **kwargs):
    """Apply module to variables and return output and modified variables."""
    new_variables = _unfreeze_variables(variables, mutable)
    with Scope(new_variables, rngs=rngs).temporary() as root:
      clone = self.clone(parent=root)
      y = getattr(clone, method)(*args, **kwargs)
    if mutable:
      return y, freeze(new_variables)
    else:
      return y

  def init_with_output(self, rngs, *args, method='__call__', **kwargs):
    """Create initialized data for module and return it with output."""
    if not isinstance(rngs, dict):
      assert rngs.shape == (2,)
      rngs = {'param': rngs}
    return self.apply(
      {}, *args, rngs=rngs, method='__call__', mutable=True, **kwargs)

  def init(self, rngs, *args, method='__call__', **kwargs):
    """Create and return initialized data for module with rngs."""
    y, v_out = self.init_with_output(rngs, *args, method=method, **kwargs)
    return v_out