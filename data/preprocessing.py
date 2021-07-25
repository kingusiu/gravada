import numpy as np


def mask_events_outliers(events, indices, values): # -> np.ndarray
    ''' remove outliers from dataset 
        events ... dataset
        indices ... indices of features to be cleaned
        values ... for each features: value at which to cut feature
        returns ... dataset without events with outliers
    '''
    mask = np.ones(len(events), dtype=bool)
    for (idx, val) in zip(indices, values):
        passed = np.abs(events[:,:,idx]) < val
        mask *= passed.all(axis=1)
    return events[mask]


def normalize_features(particles):
    ''' normalize dataset
        cylindrical & cartesian coordinates: gaussian norm
        pt: min-max norm

    '''
    idx_px, idx_py, idx_pz, idx_pt, idx_eta, idx_phi = range(6)
    # min-max normalize pt
    particles[:,:,idx_pt] = min_max_norm(particles, idx_pt)
    # standard normalize angles and cartesians
    for idx in (idx_px, idx_py, idx_pz, idx_eta, idx_phi):
        particles[:,:,idx] = std_norm(particles, idx)
    return particles

