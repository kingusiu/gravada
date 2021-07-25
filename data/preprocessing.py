import numpy as np


def mask_events_outliers(events, indices, values):
    mask = np.ones(len(events), dtype=bool)
    for (idx, val) in zip(indices, values):
        passed = np.abs(events[:,:,idx]) < val
        mask *= passed.all(axis=1)
    return events[mask]
