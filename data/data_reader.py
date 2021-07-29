import h5py
import numpy as np

import pofah.util.event_sample as evsa
import pofah.phase_space.cut_constants as cuts


def read_event_samples_from_file(filepath, all_jets=False):

    ''' reads event based inputs into object input list excluding pf candidates of jets 
        input: full path to input file
        returns [N x P x 6] numpy array where
            N ... number of events
            P ... number of objects (excluding pf candidates)
            6 ... features: px, py, pz, pt, eta, phi
    '''

    object_types = ['Ele','FatJet','Jet','Muon','Pho'] if all_jets else ['Ele','Jet','Muon','Pho']

    obj_all_feat_list = []

    with h5py.File(filepath, 'r') as ff:

        for obj in object_types:
            obj_cart = np.array(ff.get(obj+'_cart'))
            obj_cyl = np.array(ff.get(obj+'_cyl'))
            obj_all_feat = np.concatenate((obj_cart, obj_cyl), axis=-1)
            obj_all_feat_list.append(obj_all_feat)

    return np.concatenate(obj_all_feat_list, axis=1).astype('float32')


def read_dijet_samples_from_file(file_path, sample_id='qcd_side'):
    sample_cuts = cuts.sideband_cuts if sample_id == 'qcd_side' else cuts.signalregion_cuts
    sample = evsa.EventSample.from_input_file(sample_id, file_path, **sample_cuts)
    consti_j1, consti_j2 = sample.get_particles()
    samples = np.vstack([consti_j1, consti_j2])
    np.random.shuffle(samples)
    return samples

