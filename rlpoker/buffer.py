import abc
from collections import deque
import typing

import numpy as np


class Buffer:

    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)

    @abc.abstractmethod
    def append(self, item):
        """Adds the item to the buffer."""

    def sample(self, n: int, replace: bool=True):
        """Samples n items uniformly from the buffer.

        Args:
            n: int. Number of items to sample.
            replace: bool. Whether to sample with or without replacement.
        """
        indices = list(np.random.choice(self.__len__(), n, replace=replace))
        return [self.buffer[i] for i in indices]

    def get_elements(self, indices: typing.List[int]) -> typing.List[typing.Any]:
        """Returns the elements at the given indices in the deque.

        Args:
            indices: list of ints. The indices of the elements to return.

        Returns:
            list of the elements at those indices.
        """
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class Reservoir(Buffer):
    def __init__(self, maxlen):
        self.i = 0

        super().__init__(maxlen)

    def append(self, item):
        """Implements reservoir sampling.

        Let the item be the ith item. If i < self.maxlen, then we keep the item. Otherwise, we keep the new item with
        probability self.maxlen / i and otherwise discard it. If we keep the new item, we randomly choose an old item
        to discard.
        """
        self.i += 1
        if self.__len__() < self.maxlen:
            self.buffer.append(item)
        else:
            # With probability self.maxlen / i, replace an existing item with the new item.
            if np.random.rand() < self.maxlen / self.i:
                discard_idx = np.random.choice(self.maxlen)
                self.buffer[discard_idx] = item

    def __repr__(self):
        return "Reservoir(buffer={})".format(self.buffer)


class CircularBuffer(Buffer):
    """Implements a circular buffer with maximum length.
    """

    def __init__(self, maxlen=None):
        super().__init__(maxlen)

    def append(self, item):
        """Appends an item to the buffer.
        """
        self.buffer.append(item)

    def __repr__(self):
        return "CircularBuffer(buffer={})".format(self.buffer)
