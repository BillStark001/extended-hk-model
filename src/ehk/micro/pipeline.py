import time
from inspect import signature, Parameter
from typing import Callable, Dict, Any, List, Optional, Union, Tuple, Iterable, TypeAlias

import networkx as nx

from ehk.common.introspection import analyze_last_return

StrDepList: TypeAlias = str | Tuple[str, ...] | List[str]


class Context:
  def __init__(self, ignore_prefix: Optional[str] = None) -> None:
    self.ignore_prefix = ignore_prefix
    self.state: Dict[str, Any] = {}

    # { [return value name / multi selector name]: (dependencies, selector function) }
    self.selectors: Dict[str, Tuple[Tuple[str, ...], Callable]] = {}
    # { [return value name]: (selector name, index_of_ret) }
    self.multi_selectors: Dict[str, Tuple[str, int]] = {}
    self.cache: Dict[str, Any] = {}
    self._dep_graph: nx.DiGraph | None = None
    self._dep_cyclic = False

    self.debug = False
    self.debug_time_threshold = 0.01

  def set_state(self, **state: Any) -> None:
    for key, value in state.items():
      self.state[key] = value
    self.cache = {}

  def clear_cache(self):
    self.cache = {}

  def add_selector(
      self,
      name: Union[str, Tuple[str, ...]],
      deps: List[str],
      func: Callable
  ) -> None:
    if isinstance(name, tuple):
      name_tuple = name
      selector_name = str(name_tuple)

      if len(set(name_tuple)) != len(name_tuple):
        raise ValueError(
            f"Multi-Selector '{selector_name}' contains duplicate return names")

      # Validate first so registration is atomic on failure.
      for n in name_tuple:
        if n in self.multi_selectors or n in self.selectors:
          raise ValueError(
              f"Multi-Selector '{selector_name}' is already defined in the context")
      if selector_name in self.multi_selectors or selector_name in self.selectors:
        raise ValueError(
            f"Selector '{selector_name}' is already defined in the context")

      # Commit writes only after all checks pass.
      for i, n in enumerate(name_tuple):
        self.multi_selectors[n] = (selector_name, i)
      name = selector_name
    if name in self.multi_selectors or name in self.selectors:
      raise ValueError(f"Selector '{name}' is already defined in the context")
    self.selectors[name] = (tuple(deps), func)

    # invalidate graph cache
    self._dep_graph = None
    self._dep_cyclic = False

  def get_dep_graph(self):
    G = nx.DiGraph()
    for sel, (deps, _) in self.selectors.items():
      for dep in deps:
        G.add_edge(dep, sel)
    for sel, (msel, _) in self.multi_selectors.items():
      G.add_edge(msel, sel)

    # calculate states
    state_vars: List[str] = [
        node for node in G.nodes() if G.in_degree(node) == 0]
    cycles: List[List[str]] = list(nx.simple_cycles(G))

    return G, state_vars, cycles

  def get_state_names(self) -> List[str]:
    return list(self.selectors.keys()) + list(self.multi_selectors.keys())

  def clone(self, state=False):
    c = Context(self.ignore_prefix)
    c.selectors = dict(**self.selectors)
    c.multi_selectors = dict(**self.multi_selectors)
    if state:
      c.state = dict(**self.state)
    return c

  def get(self, name: str) -> Any:
    if name in self.state:
      return self.state[name]
    if name in self.cache:
      return self.cache[name]
    if name in self.multi_selectors:
      selector_key, index = self.multi_selectors[name]
      ret = self.get(selector_key)
      return ret[index]
    if name not in self.selectors:
      raise ValueError(
          f"Selector '{name}' is not defined in the context")

    deps, func = self.selectors[name]
    dep_values = [self.get(dep) for dep in deps]
    value = func(*dep_values)
    self.cache[name] = value
    return value

  def get_values(self, *names):
    return [self.get(n) for n in names]

  def invalidate(self, name: str, recursive=True, multi=False) -> None:
    if name in self.state:
      del self.state[name]
    if name in self.cache:
      del self.cache[name]

    if not recursive:
      return

    # get or create graph
    if self._dep_graph is None:
      self._dep_graph, _, _c = self.get_dep_graph()
      self._dep_cyclic = len(_c) > 0

    # if name is from multi selector,
    # invalidate the multi selector itself
    if multi and name in self.multi_selectors:
      name = self.multi_selectors[name][0]

    # find nodes and delete them
    nodes_to_delete: Iterable[str] = nx.dfs_preorder_nodes(
        self._dep_graph,
        name,
        len(self.selectors) + len(self.multi_selectors)
    )
    for n in nodes_to_delete:
      if n in self.cache:
        del self.cache[n]

  def __getattr__(self, name: str) -> Any:
    if self.debug:
      tstart = time.time()
    val = self.get(name)
    if self.debug:
      tdelta = time.time() - tstart
      if tdelta > self.debug_time_threshold:
        print(f'Evaluation of state "{name}" costs {tdelta}s')

    return val

  def __delattr__(self, name: str) -> None:
    internal_attrs = {
        'ignore_prefix',
        'state',
        'selectors',
        'multi_selectors',
        'cache',
        '_dep_graph',
        '_dep_cyclic',
        'debug',
        'debug_time_threshold',
    }

    # Keep normal object semantics for internal fields.
    if name in internal_attrs:
      object.__delattr__(self, name)
      return

    # `del context.xxx` is treated as state invalidation.
    if name in self.state or name in self.cache or name in self.selectors or name in self.multi_selectors:
      self.invalidate(name, recursive=True, multi=True)

  def selector(
      self,
      return_name: None | Callable | StrDepList = None,
      deps_map: Optional[Dict[str, str]] = None
  ) -> Callable:

    output_var: None | StrDepList = None

    def decorator(func: Callable) -> Callable:
      if not callable(func):
        raise TypeError(
            'Decorator applied on a non-callable object: ' + str(func))
      sig = signature(func)
      params = sig.parameters

      input_vars = [
          name for name, param in params.items()
          if param.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
      ]

      # auto parse return name if not assigned
      nonlocal output_var
      if output_var is None:
        output_var = analyze_last_return(func)
        if output_var is None:
          output_var = func.__name__

      # ensure it is a string tuple with length more than 2 or a string
      if isinstance(output_var, list):
        output_var = tuple(output_var)
      if isinstance(output_var, tuple) and len(output_var) == 1:
        output_var = output_var[0]

      # ignore_prefix
      if self.ignore_prefix \
          and isinstance(output_var, str) \
              and output_var.startswith(self.ignore_prefix):
        output_var = output_var[len(self.ignore_prefix):]

      if deps_map:
        input_vars = [deps_map.get(name, name) for name in input_vars]

      self.add_selector(output_var, input_vars, func)
      return func

    # for usage `@context.selector`
    if callable(return_name):
      return decorator(return_name)
    else:
      output_var = return_name

    # for usage `@context.selector()`
    return decorator


if __name__ == "__main__":
  context = Context()

  context.set_state(a=10, b=20)

  @context.selector(return_name='sum', deps_map=dict(x='a', y='b'))
  def c(x: int, y: int) -> int:
    return x + y

  @context.selector('double')
  def d(sum: int) -> int:
    return sum * 2

  @context.selector(('e', 'f'), deps_map=dict(c='sum', d='double'))
  def ef(c: int, d: int):
    return c + d, c - d

  print(context.double)
  print(context.sum)
  print(context.e)
  print(context.f)
