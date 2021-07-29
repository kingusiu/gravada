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


def std_norm(data, idx):
    return (data[:,:,idx] - np.mean(data[:,:,idx]))/np.std(data[:,:,idx])


def min_max_norm(data, idx):
    return (data[:,:,idx] - np.mean(data[:,:,idx]))/np.std(data[:,:,idx])


def normalize_features(particles, feature_names):
    ''' normalize dataset
        cylindrical & cartesian coordinates: gaussian norm
        pt: min-max norm

    '''
    # min-max normalize pt
    idx_pt = feature_names.index('pt')
    particles[:,:,idx_pt] = min_max_norm(particles, idx_pt)
    # standard normalize angles and cartesians
    for idx, _ in enumerate([n for n in feature_names if 'pt' not in feature_names]):
        particles[:,:,idx] = std_norm(particles, idx)
    return particles

