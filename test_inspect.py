import ast
import inspect

source = """
def {funcname}(obsset, {kwargs_list}):
    print(arg1)
    pass
"""

# source = inspect.getsource(myf)

tree = ast.parse(source)
