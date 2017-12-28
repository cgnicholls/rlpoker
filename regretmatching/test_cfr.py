import numpy as np
from cfr import compute_regret_matching, compare_strategies

def test_compute_regret_matching():
	assert compute_regret_matching({1: 3.0, 2: 5.0, 3: 0.0}) == {1: 3.0/8.0, 2: 5.0/8.0, 3: 0.0}
	assert compute_regret_matching({1: 0.0, 2: -1.0}) == {1: 0.5, 2: 0.5}

def test_compare_strategies():
	s1 = {1: {3: 2.0, 4: 3.0}, 2: {1: 1.0, 2: 13.0}}
	s2 = {1: {3: 3.0, 4: 5.0}, 2: {1: 2.0, 2: 13.5}}

	expected = 0.5 * np.sqrt((2.0 - 3.0)**2 + (3.0 - 5.0)**2) + 0.5 * np.sqrt((1.0 - 2.0)**2 + (13.0 - 13.5)**2)
	computed = compare_strategies(s1, s2)

if __name__ == "__main__":
	test_compute_regret_matching()
	test_compare_strategies()