"""
* Storage classes are only parameters of Source node

Dependency framework is based on comparing timestamps, rather than marking nodes as "dirty". With the latter method
you could not handle invalidation of untouched branches (everything that is not in dependency tree of selected node).
You could simulate time-less invalidation by a pseudo count/timestamp.

TODO:
* Exceptions?
* What if self.dependency_nodes empty?
* "is_dirty" indicator in output? is_dirty() function?
"""

from .depnodes import MemCache, Operator, MultiOutputProxy, Source
from .storage import FileStorage, MemStorage


def dependency_operator(output_num=None):
    """
    Decorator to make any function a cached Operator
    Due to the lazy nature, it needs to know the number of potentially multiple outputs in advance.
    output_num equal to an integer will create MultiOutputProxy if you write `x,y=f(..)`

    Note that decorated functions will return a DependencyNode and data needs to be accessed by `.data`

    >>> @dependency_operator()
    >>> def f(..):
    >>>     return y
    >>> y = f(x)              # y will be lazy DependencyNode, use y.data to get actual data

    Support of lazy multi-output, e.g.
    >>> @dependency_operator(2)
    >>> def f(..):
    >>>     return x, y
    >>> x, y = f(..)        # x, y will be lazy DependencyNodes
    """
    if output_num is None:
        def wrapper(func):
            def partial_cached_operator(*args, **kwargs):
                return MemCache(Operator(func, *args, **kwargs))

            return partial_cached_operator

        return wrapper
    else:
        def wrapper_multi(func):
            def partial_cached_operator_multi(*args, **kwargs):
                data_cache = MemCache(Operator(func, *args, **kwargs))
                return tuple(MultiOutputProxy(data_cache, i) for i in range(output_num))

            return partial_cached_operator_multi

        return wrapper_multi


def file_source(filename):
    return Source(FileStorage(filename))


def constant_source(data):
    return Source(MemStorage(data, name=str(data)))
