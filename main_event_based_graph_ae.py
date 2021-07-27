#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import setGPU
from importlib import reload
import mplhep as hep

import data.data_reader as dare
import data.preprocessing as prep
import data.graph_construction as grctr
import models.graph_nn as grap
import analysis.plotting as plott


# In[21]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # read in data

# In[22]:


input_dir = '/eos/cms/store/group/ml/AnomalyHackathon/VAE/samples/v1/qcd/merged/'
file_name = 'flat_scouting_2_numEvent500_event.h5'
file_path = os.path.join(input_dir, file_name)


# In[23]:


samples = dare.read_event_samples_from_file(file_path)


# In[24]:


#samples = samples[:2000]


# In[25]:


samples.shape


# In[26]:


features_names = ['px', 'py', 'pz', 'pt', 'eta', 'phi']


# In[27]:


# mask events with ojects having pt > 13000
reload(prep)
pt_idx = features_names.index('pt')
pz_idx = features_names.index('pz')
samples = prep.mask_events_outliers(samples, [pt_idx, pz_idx], [5e3, 5e3])


# In[28]:


samples.shape


# ### sparsity

# In[29]:


particles_n = samples.shape[0]*samples.shape[1]
true_particles_n = sum(samples[:,:,pt_idx].flatten() > 0)
print('num total particles ', particles_n)
print('num true particles', true_particles_n)
print('true particles share ', (float(true_particles_n)/float(particles_n))*100.0)


# ## normalize

# In[30]:


reload(prep)
samples = prep.normalize_features(samples)


# In[31]:


for i in range(samples.shape[2]):
    plt.figure()
    _ = plt.hist(samples[:,:,i].flatten(), bins=100)
    plt.title(features_names[i])
    plt.yscale('log')


# # prepare input for model

# In[32]:


nodes_n = samples.shape[1]
feat_sz = samples.shape[2]


# In[33]:


batch_size = 128


# In[34]:


reload(grctr)
A = grctr.make_adjacencies(samples, pt_idx=pt_idx)
A_tilde = grctr.normalized_adjacency(A)


# # build and train model

# In[35]:


reload(grap)
gnn = grap.GraphAutoencoder(nodes_n=nodes_n, feat_sz=feat_sz, activation=tf.nn.selu, graphLayer='convExt')
gnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), run_eagerly=True)


# In[36]:


callbacks = [tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, verbose=1)]
gnn.fit((samples, A_tilde), A, epochs=100, batch_size=128, validation_split=0.25, callbacks=callbacks)


# ## predict background

# In[37]:


input_dir_qcd_test = '/eos/cms/store/group/ml/AnomalyHackathon/VAE/samples/v1/qcd/merged'
file_name_qcd_test = 'flat_scouting_3_numEvent500_event.h5'
file_path_qcd_test = os.path.join(input_dir_qcd_test, file_name_qcd_test)


# In[38]:


samples_qcd_test = dare.read_event_samples_from_file(file_path_qcd_test)
samples_qcd_test = prep.normalize_features(samples_qcd_test)
samples_qcd_test = samples_qcd_test[:3000]
print(samples_qcd_test.shape)


# In[39]:


A_qcd_test = grctr.make_adjacencies(samples_qcd_test, pt_idx=pt_idx)
A_qcd_test_tilde = grctr.normalized_adjacency(A_qcd_test)


# In[40]:


z_qcd, A_qcd_pred = gnn((samples_qcd_test, A_qcd_test_tilde))
#A_qcd_reco = (tf.nn.sigmoid(A_qcd_pred) > 0.5).numpy().astype('int')


# In[41]:


loss_qcd = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(A_qcd_test, A_qcd_pred), axis=(1,2)).numpy()


# ## predict signal

# In[42]:


# signal dict
signal_dict = {
    
    'Zprime1000' : 'ZprimeToZH_MZprime1000_MZ50_MH80_narrow',
    'Graviton' : 'BulkGraviton_hh_GF_HH',
    'SMS' : 'SMS-T1qqqq'
    
}

input_dir_signal = '/eos/cms/store/group/ml/AnomalyHackathon/VAE/samples/v1'


# In[43]:


signal_losses = []

for sig_id in signal_dict.keys():
    
    # read signal files
    file_num = np.random.randint(9)
    file_name = 'flat_scouting_' + str(file_num) + '_numEvent500_event.h5'
    file_path_signal = os.path.join(input_dir_signal, signal_dict[sig_id], 'merged', file_name)
    print('predicting ', file_path_signal)
    samples_signal_test = dare.read_event_samples_from_file(file_path_signal)
    
    # preprocess
    samples_signal_test = prep.mask_events_outliers(samples_signal_test, [pt_idx, pz_idx], [5e3, 5e3])
    samples_signal_test = prep.normalize_features(samples_signal_test)
    samples_signal_test = samples_signal_test[:10000]
    print(samples_signal_test.shape)
    
    # make adjacencies
    A_signal_test = grctr.make_adjacencies(samples_signal_test, pt_idx=pt_idx)
    A_signal_test_tilde = grctr.normalized_adjacency(A_signal_test)
    
    # run inference
    z_signal, A_signal_pred = gnn((samples_signal_test, A_signal_test_tilde))
    #A_signal_reco = (tf.nn.sigmoid(A_signal_pred) > 0.5).numpy().astype('int')
    loss_signal = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(A_signal_test, A_signal_pred), axis=(1,2)).numpy()
    signal_losses.append(loss_signal)


# # plot loss distributions and ROC

# In[44]:


loss_qcd.shape
print(min(loss_qcd))


# In[45]:


plt.style.use(hep.style.CMS)
_ = plt.hist([loss_qcd.flatten(), *signal_losses], bins=100, label=['qcd']+list(signal_dict.keys()), histtype='step', density=True)
plt.yscale('log')
plt.legend()


# In[46]:


print(min(loss_qcd), max(loss_qcd))


# In[47]:


print(min(loss_signal), max(loss_signal))


# In[48]:


len(signal_losses)


# In[49]:


reload(plott)
plott.plot_roc([loss_qcd]*3, signal_losses, legend=list(signal_dict.keys()), log_x=False)


# In[ ]:




