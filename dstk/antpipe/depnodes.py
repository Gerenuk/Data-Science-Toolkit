import logging
from abc import ABC, abstractmethod
from collections import namedtuple

logger = logging.getLogger(__name__)

TreeShowNode = namedtuple(
    "TreeShowNode", "name children"
)  # Used to build the (abstract) tree structure for printing


class DependencyNode(ABC):
    """
    Dependency nodes are detect as subclasses from this class, rather than duck-typing for attributes.
    """

    @abstractmethod
    def _data(self):
        """
        return actual data (without update)
        """
        pass

    @abstractmethod
    def _timestamp(self):
        """
        return timestamp
        """
        pass

    @abstractmethod
    def _update(self):
        """
        unconditionally update stored data
        """
        pass

    @abstractmethod
    def _tree_show_node(self):
        """
        return data structure for plotting the dependency graph. needs at least .name, .children
        """
        pass

    @property
    def data(self):
        """
        does update and returns current data
        """
        self._update()
        return self._data


class Source(DependencyNode):
    def __init__(self, storage):
        self.storage = storage
        self.name = self.storage.name

    @property
    def _data(self):
        logger.info("Reading source {}".format(self.name))
        return self.storage.read()

    @property
    def _timestamp(self):
        return self.storage.timestamp

    def _update(self):
        pass

    def _tree_show_node(self):
        return TreeShowNode("SRC  {}".format(self.name), [])


class MultiOutputProxy(DependencyNode):
    def __init__(self, reference, output_index):
        self.reference = reference
        self.output_index = output_index

    @property
    def _data(self):
        return self.reference._data[self.output_index]

    @property
    def _timestamp(self):
        return self.reference._timestamp

    def _update(self):
        self.reference._update()

    def _tree_show_node(self):
        return "?"  # TODO


class Operator(DependencyNode):
    """
    Uncached
    """

    def __init__(self, func, *inputs):
        self.func = func
        self.inputs = inputs

        self.name = self.func.__name__
        self.dependency_nodes = [
            inp for inp in self.inputs if isinstance(inp, DependencyNode)
        ]

        assert self.dependency_nodes  # TODO: What if empty?

    @property
    def _data(self):
        logger.info("Executing operator {}".format(self.name))
        return self.func(
            *[
                inp._data if isinstance(inp, DependencyNode) else inp
                for inp in self.inputs
            ]
        )

    @property
    def _timestamp(self):
        return max([inp._timestamp for inp in self.dependency_nodes])

    def _update(self):
        for inp in self.dependency_nodes:
            inp._update()

    def _tree_show_node(self):
        return TreeShowNode(
            "FUNC {}".format(self.name),
            [inp._tree_show_node() for inp in self.dependency_nodes],
        )


class MemCache(DependencyNode):
    EMPTY = object()

    def __init__(self, input_):
        self.input = input_
        self.mem_data = self.EMPTY
        self.mem_timestamp = self.EMPTY
        self.name = "Cache"  # TODO: better name

    @property
    def _data(self):
        assert self.mem_data is not self.EMPTY
        logger.info("Returning value from cache {}".format(self.name))
        return self.mem_data

    @property
    def _timestamp(self):
        assert self.mem_timestamp is not self.EMPTY
        return self.mem_timestamp

    def _update(self):
        self.input._update()

        if self.mem_timestamp is self.EMPTY:
            self._update_initial_cache_data()
        elif self.input._timestamp > self.mem_timestamp:
            self._update_outdated_cache_data()

    def _update_initial_cache_data(self):
        logger.info("Loading initial cache {}".format(self.name))
        self.mem_data = self.input._data
        self.mem_timestamp = self.input._timestamp

    def _update_outdated_cache_data(self):
        logger.info("Updating outdated cache {}".format(self.name))
        self.mem_data = self.input._data
        self.mem_timestamp = self.input._timestamp

    def _tree_show_node(self):
        source_tree_show_node = self.input._tree_show_node()

        if self.mem_timestamp is not self.EMPTY:
            return TreeShowNode("CCH  {}".format(self.name), [source_tree_show_node])
        else:
            return TreeShowNode(
                "CCH{} {}".format(" ", self.name), [source_tree_show_node]
            )


class StorageCache(MemCache):
    def __init__(self, input_, storage):
        super().__init__(input_)
        self.storage = storage
        self.name = self.storage.name
        try:
            self.mem_timestamp = storage.timestamp
        except FileNotFoundError:  # TODO: too specific for a file system?
            self.mem_data = self.EMPTY  # TODO: working?

    def _update_initial_cache_data(self):  # TODO: read file cache or recalc?
        logger.info("Reading cache {}".format(self.name))
        self.mem_data = self.storage.read()
        self.mem_timestamp = self.storage.timestamp

    def _update_outdated_cache_data(self):
        super()._update_cache_data()
        logger.info("Writing to storage")
        self.storage.write(self.mem_data)
