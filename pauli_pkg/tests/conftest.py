import random, numpy as np

def pytest_configure():
    # Give deterministic failures but still different random circuits each run
    random.seed()                # system time
    np.random.seed()
