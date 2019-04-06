from collections import deque

from rlpoker.buffer import CircularBuffer, Reservoir


def test_circular_buffer_max_len():
    buffer = CircularBuffer(maxlen=10)

    for i in range(11):
        buffer.append(i)

    assert buffer.buffer == deque(list(range(1, 11)))

    samples = buffer.sample(5)

    assert set(samples).issubset(set(range(1, 11)))

    assert buffer.buffer == deque(list(range(1, 11)))

    buffer.append(11)

    assert buffer.buffer == deque(list(range(2, 12)))


def test_reservoir():
    buffer = Reservoir(maxlen=5)

    for i in range(5):
        buffer.append(i)
        assert buffer.buffer == deque(list(range(i+1)))

    buffer.append(5)

    samples = buffer.sample(4)
    assert set(samples).issubset(set(range(6)))

    items = set(buffer.buffer)
    possible_items = set(range(6))
    assert items.issubset(possible_items)
