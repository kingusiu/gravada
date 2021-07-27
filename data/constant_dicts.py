sample_dict = {
    
    'feature_names' : {
        'events' : ['px', 'py', 'pz', 'pt', 'eta', 'phi'],
        'dijet' : ['eta', 'phi', 'pt'],
        'dijet_augmented' : ['eta', 'phi', 'pt', 'px', 'py', 'pz']
        },

    'dir_paths': {
        'events' : {
            'qcd' : '/eos/cms/store/group/ml/AnomalyHackathon/VAE/samples/v1/qcd/merged/',
            'grs' : '/eos/cms/store/group/ml/AnomalyHackathon/VAE/samples/v1/BulkGraviton_hh_GF_HH/merged',
            'zprime' : '/eos/cms/store/group/ml/AnomalyHackathon/VAE/samples/v1/ZprimeToZH_MZprime1000_MZ50_MH80_narrow/merged',
            'sms' : '/eos/cms/store/group/ml/AnomalyHackathon/VAE/samples/v1/SMS-T1qqqq/merged',
        },
        'dijet' : {
            'qcd_side' : '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_sideband_parts',
            'qcd_sig' : '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_signalregion_parts',
            'grs35na' : '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/RSGraviton_WW_NARROW_13TeV_PU40_3.5TeV_NEW_parts',
            'grs35' : '/eos/project/d/dshep/TOPCLASS/DijetAnomaly/VAE_data/events/RSGraviton_WW_BROAD_13TeV_PU40_3.5TeV_NEW_parts',
        },
    }
}
