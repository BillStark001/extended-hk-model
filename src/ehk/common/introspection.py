from typing import Callable

import inspect
import ast


def strip_leading_spaces(source: str):
  space_count = len(source) - len(source.lstrip(' \t'))
  if space_count > 0:
    source_splitted = [(
        x[space_count:] if len(x) > space_count else ''
    ) for x in source.split('\n')]
    source = '\n'.join(source_splitted)
  return source


def analyze_last_return(func: Callable):

  source = inspect.getsource(func)
  source = strip_leading_spaces(source)
  tree = ast.parse(source, mode='exec')
  function_def = next(node for node in ast.walk(
      tree) if isinstance(node, ast.FunctionDef))

  last_return = None
  for node in function_def.body:
    if isinstance(node, ast.Return):
      last_return = node

  if last_return is None:
    return None

  return_value = last_return.value

  if isinstance(return_value, ast.Name):
    return return_value.id
  elif isinstance(return_value, ast.Tuple):
    if all(isinstance(elt, ast.Name) for elt in return_value.elts):
      return tuple(elt.id for elt in return_value.elts)  # type: ignore

  return None


if __name__ == "__main__":
  def test_function():
    def nested_test_function():
      a = 3
      return a
    x = 1
    y = 2
    return x, y

  result = analyze_last_return(test_function)
  print(result)
