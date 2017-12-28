import numpy as np
from cfr import compute_regret_matching

def test_compute_regret_matching():
	assert compute_regret_matching({1: 3.0, 2: 5.0, 3: 0.0}) == {1: 3.0/8.0, 2: 5.0/8.0, 3: 0.0}
	assert compute_regret_matching({1: 0.0, 2: -1.0}) == {1: 0.5, 2: 0.5}

if __name__ == "__main__":
	test_compute_regret_matching()