#import neccessary libraries
import numpy as np
import evaluation as ev
import csv
import pandas as pd

#modified test_single_photon function (taken from test_stats.py) that returns number of clusters
#this function calls our model and tests it using the root file defined in datasets.py in the single_photon_dataset function
#modifications w.r.t evaluations.py:
#-added n_events & noise variables to define number of events & singal threshold each time we call our function
#-added n_clusters variable which gets the length of our clustering array, thus our number of clusters
#-deleted plotly data functions to save compilation time
def n_clusters(n_events, noise):
    tbeta = .2
    td = .5
    nmax = n_events
    yielder = ev.TestYielderSinglePhoton()
    yielder.model.signal_threshold = noise
    n_clusters = np.zeros(shape=(nmax))
    for i, (clustering) in enumerate(yielder.iter_clustering(tbeta, td, nmax=nmax)):
        n_clusters[i] = len(np.unique(clustering))
    return n_clusters
    
#modifiedtest_single_photon function (taken from test_stats.py) that returns cluster per hit & energy per hit
#this function calls our model and tests it using the root file defined in datasets.py in the single_photon_dataset function
def clusters_info(n_events, noise):
    tbeta = .2
    td = .5
    nmax = n_events
    root_file = 'step3_Gamma_E25_n1000_part9.root' 
    yielder = ev.TestYielderSinglePhoton()
    yielder.model.signal_threshold = noise
    event_n = []
    event_cluster = []
    cluster_energy = []
    cluster_nhits = []
    for i, (event, prediction, clustering) in enumerate(yielder.iter_clustering(tbeta, td, nmax=nmax)):
        #see https://stackoverflow.com/questions/67108215/how-to-get-sum-of-values-in-a-numpy-array-based-on-another-array-with-repetitive
        np.set_printoptions(suppress=True) #suppress scientific notation
        _, idx, _ = np.unique(clustering, return_counts=True, return_inverse=True)
        energy_values = np.round(np.bincount(idx, event.energy),3)
        event_cluster_unique, n_hits = np.unique(clustering, return_counts=True, return_index=False)
        event_i = np.full(len(event_cluster_unique), i, dtype=int)
        event_n = np.append(event_n, event_i)
        event_cluster = np.append(event_cluster, event_cluster_unique)
        cluster_nhits = np.append(cluster_nhits, n_hits)
        cluster_energy = np.append(cluster_energy, energy_values)
    return event_n, event_cluster, cluster_nhits, cluster_energy
    
#set run parameters for cluster-noise distribution
n_events = 100
noise_range = range(1,20)
noise_step = 0.05
n_clusters_mean = np.zeros(shape=(len(noise_range)))

#create .csv file to write our results on for cluster-noise distribution
with open('cluster_noise_distribution-test.csv', 'w') as f:
    fieldnames = ['noise', 'n_events', 'n_clusters_mean']
    writer = csv.DictWriter(f,fieldnames=fieldnames)
    writer.writeheader()
    #call our function and loop over incremental noise values (step set by noise_step)
    for noise_i in noise_range:
        noise = np.round(noise_i*noise_step,2)
        n_clusters = test_single_photon_n_clusters(n_events, noise)
        n_clusters_mean[noise_i] = np.round(np.mean(n_clusters), 2)
        writer.writerow({'noise': noise, 'n_events': n_events, 'n_clusters_mean': n_clusters_mean[noise_i]})
        print(noise_i, noise, n_events, n_clusters, n_clusters_mean[noise_i])

#read results as pandas dataframe        
df = pd.read_csv('cluster_noise_distribution-test.csv')

#plots cluster-noise distribution results & save as .png
ymin = 0
ymax = np.int(df['n_clusters_mean'].max()) + 5

ax = df.plot('noise', 'n_clusters_mean', 
        kind='bar', 
        title = 'Number of Clusters vs Noise Distribution for photons',
        ylabel = 'Average Number of clusters', 
        xlabel='Signal Threshold',
        rot = 45,
        legend = None,
        figsize=(6, 4))

ax.set_axisbelow(True)
ax.set_ylim([ymin, ymax])
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.get_figure().savefig('cluster_noise_distribution-test.png', bbox_inches = 'tight')

#get cluster info for n_events with the noise value where avg number of clusters is minimized
n_events = 100
noise_min = df['noise'][df['n_clusters_mean'].idxmin()]
event_n, event_cluster, cluster_nhits, cluster_energy = clusters_info(n_events,noise_min)

#create pandas dataframe with resuls from cluster info
ci = event_n, event_cluster, cluster_nhits, cluster_energy
df_ci = pd.DataFrame({'event_n': ci[0], 'event_clusters': ci[1], 'n_hits': ci[2], 'energy_values': ci[3]})

#calculate percentage of energy by cluster, per event
df_ci['energy_percentage'] = df_ci['energy_values'] / df_ci.groupby('event_n')['energy_values'].transform('sum')

#plots number of htis & save as .png
ymin = 0
ymax = np.int(df_ci['n_hits'].max()) + 5

ax = df_ci[df_ci['event_n']==0.].plot('event_clusters', 'n_hits', 
        kind='bar', 
        title = '# of hits per cluster',
        ylabel = '# of hits', 
        xlabel='cluster',
        rot = 45,
        legend = None,
        figsize=(6, 4))

ax.set_axisbelow(True)
ax.set_ylim([ymin, ymax])
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.get_figure().savefig('cluster_info_nhits-test.png', bbox_inches = 'tight')

#plots energy values & save as .png
ymin = 0
ymax = 1
ax = df_ci[df_ci['event_n']==0.].plot('event_clusters', 'energy_percentage', 
        kind='bar', 
        title = 'Total energy percentage per cluster',
        ylabel = 'Percentage of Energy (%)', 
        xlabel='cluster',
        rot = 45,
        legend = None,
        figsize=(6, 4))

ax.set_axisbelow(True)
ax.set_ylim([ymin, ymax])
ax.yaxis.grid(color='gray', linestyle='dashed')
ax.get_figure().savefig('cluster_info_energy-test.png', bbox_inches = 'tight')
