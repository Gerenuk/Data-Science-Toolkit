from math import log, ceil
import sys
import base64


class HashCollision(Exception):
    pass


class Labels:
    def __init__(self, num_elements=1000):
        self.params = {}
        self.num_chars = ceil(2 * log(num_elements) / log(64)) + 1

    def __call__(self, param):
        """
        :param param: hashable data object
        :return: string hash
        """
        label = self._label(param)
        if label not in self.params:
            self.params[label] = param
            return label

        if self.params[label] != param:
            raise HashCollision(repr(param))

        return label

    def __getitem__(self, label):
        """
        :param label: string hash previously generated
        :return: data object that was stored for this hash
        """
        return self.params[label]

    def _label(self, value):
        hash_ = hash(value)
        bytes_ = hash_.to_bytes(sys.hash_info.width // 8, "big")
        base64_ = base64.b64encode(bytes_)
        return base64_[: self.num_chars].decode("utf8")
