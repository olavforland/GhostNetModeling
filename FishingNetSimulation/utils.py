import os
import random
import numpy as np
from datetime import timedelta

def random_date(start, end):
    """
    Helper to get a random date between two datetimes
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_sim_results(filepath: str):
    with np.load(filepath) as data:
        init_lons = data['init_lons']
        init_lats = data['init_lats']
        lon_active = data['lon_active']
        lat_active = data['lat_active']
        lon_deactive = data['lon_deactive']
        lat_deactive = data['lat_deactive']
    return init_lons, init_lats, lon_active, lat_active, lon_deactive, lat_deactive

