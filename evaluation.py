import torch
from torch_geometric.data import DataLoader
import numpy as np
import tqdm  #(te quiero demasiado), creates a loading bar for iterable objects
from time import strftime
import os, os.path as osp
import uuid #(Universally Unique Identifier), creates and assigns unique IDs to objects

from datasets import tau_dataset, single_photon_dataset
from torch_cmspepr.gravnet_model import GravnetModelWithNoiseFilter
import torch_cmspepr.objectcondensation as oc

from matching import match, group_matching
from colorwheel import ColorWheel, HighlightColorwheel, ColorwheelWithProps

import warnings
warnings.filterwarnings("ignore")


#perform principal component analysis to reduce cluster space dimensions
def pca_down(cluster_space_coords: np.array, n_components: int = 3):
    from sklearn.decomposition import PCA
    dim = cluster_space_coords.shape[1]
    if dim <= n_components: return cluster_space_coords
    pca = PCA(n_components)
    out = pca.fit_transform(cluster_space_coords)
    assert out.shape == (cluster_space_coords.shape[0], n_components)
    return out

#load model and model weights
def get_model():
    #if signal_threshold is None then signal_threshold = 0.15
    #model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=.05)
    model = GravnetModelWithNoiseFilter(input_dim=9, output_dim=6, k=50, signal_threshold=0.15)
    ckpt = 'ckpt_train_taus_integrated_noise_Oct20_212115_best_397.pth.tar'
    model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
    return model

#get tau dataset and split it randomly 80%
def get_dataset():
    _, test_dataset = tau_dataset().split(.8)
    return test_dataset

#define class test yielder
class TestYielder:
    #intiialize testyielder attributes (self, model, dataset, weights)
    def __init__(self, model=None, dataset=None, ckpt=None):
        self.model = get_model() if model is None else model
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
        self.dataset = get_dataset() if dataset is None else dataset
        self.reset_loader()

    #reset data loader
    def reset_loader(self):
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    #create event_filter (why does it need to create an empty event filter?)
    def event_filter(self, event):
        """Subclassable to make an event-level filter before any model inference (for speed)"""
        return True

    #set class to be an iterable object
    def __iter__(self):
        """
        Default iterator: Just the data
        """
        return self.iter()

    #iterates over the data and adds a loading bar using tqdm
    def _iter_data(self, nmax=None, start=None):
        i_done = 0
        for i, data in tqdm.tqdm(enumerate(self.loader), total=(len(self.loader) if nmax is None else nmax)):
            if start is not None and i < start: continue
            if nmax is not None and i_done == nmax: break
            yield i, data
            i_done += 1

    #iterates over the events in the data; continues until it finds a non-filtered event (i.e not noise) and saves the event for later 
    def iter(self, nmax=None, start=None):
        for i, data in self._iter_data(nmax, start):
            event = Event(data)
            if not self.event_filter(event): continue
            yield event

    #takes the non-filtered event and evaluates the model on it, calculateing betas, getting cluster coords and storing the predictions; saves event and prediction for later
    def iter_pred(self, nmax=None, start=None):
        with torch.no_grad():
            self.model.eval()
            for i, data in self._iter_data(nmax, start):
                event = Event(data)
                if not self.event_filter(event): continue
                _, pass_noise_filter, out_gravnet = self.model(data.x, data.batch)
                pass_noise_filter = pass_noise_filter.numpy()
                pred_betas = torch.sigmoid(out_gravnet[:,0]).numpy()
                pred_cluster_space_coords = out_gravnet[:,1:].numpy()
                prediction = Prediction(pass_noise_filter, pred_betas, pred_cluster_space_coords)
                yield event, prediction
    
    #creates clusters based on predictions
    def iter_clustering(self, tbeta, td, nmax=None, start=None):
        for event, prediction in self.iter_pred(nmax, start):
            clustering = cluster(prediction, tbeta, td)
            yield event, prediction, clustering

    #matches predictions to clusters
    def iter_matches(self, tbeta, td, nmax=None, start=None):
        for event, prediction, clustering in self.iter_clustering(tbeta, td, nmax, start):
            matches = make_matches(event, prediction, clustering=clustering)
            yield event, prediction, clustering, matches


#subclass for EM events test
class TestYielderEM(TestYielder):
    def __init__(self, *args, **kwargs): #*args: variable arguments, **kwargs: variable keywords arguments
        super().__init__(*args, **kwargs) #inherits __init__ from TestYielder
        self.min_em_fraction = 1.0

    def event_filter(self, event):
        return event.em_energy_fraction >= self.min_em_fraction

class TestYielderHAD(TestYielder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_had_fraction = 1.0

    def event_filter(self, event):
        return event.had_energy_fraction >= self.min_had_fraction

class TestYielderMIP(TestYielder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_mip_fraction = 1.0

    def event_filter(self, event):
        return event.mip_energy_fraction >= self.min_mip_fraction


class TestYielderSinglePhoton(TestYielder):
    def __init__(self, *args, **kwargs):
        kwargs['dataset'] = -1
        super().__init__(*args, **kwargs)

    def reset_loader(self):
        self.loader = single_photon_dataset()


class Event:
    def __init__(self, data):
        self.x = data.x.numpy()
        self.y = data.y.numpy()
        if hasattr(data, 'truth_cluster_props'):
            self.truth_cluster_props = data.truth_cluster_props.numpy()
        else:
            self.truth_cluster_props = np.zeros((self.x.shape[0], 5))
        if hasattr(data, 'inpz'):
            self.inpz = int(data.inpz[0].item())

    def __getitem__(self, where):
        """Selection mechanism"""
        new = self.__class__.__new__(self.__class__)
        new.x = self.x[where]
        new.y = self.y[where]
        new.truth_cluster_props = self.truth_cluster_props[where]
        if hasattr(self, 'inpz'): new.inpz = self.inpz
        return new

    #get true energy
    # assign property decorator so that the class has property attributes (fget, fset, fdel, docs)
    @property
    def truth_e_bound(self):
        return self.truth_cluster_props[:,0]

    #get true x coordinate
    @property
    def truth_x_bound(self):
        return self.truth_cluster_props[:,1]

    #get true y coordinate
    @property
    def truth_y_bound(self):
        return self.truth_cluster_props[:,2]

    #get true time
    @property
    def truth_time(self):
        return self.truth_cluster_props[:,3]

    #get true particle id
    @property
    def truth_pdgid(self):
        return self.truth_cluster_props[:,4]

    # Getters for a single truth id

    def index_by_id(self, id):
        return (self.y == id).argmax()

    def truth_e_bound_by_id(self, id):
        return self.truth_e_bound[self.index_by_id(id)]

    def truth_x_bound_by_id(self, id):
        return self.truth_x_bound[self.index_by_id(id)]

    def truth_y_bound_by_id(self, id):
        return self.truth_y_bound[self.index_by_id(id)]

    def truth_time_by_id(self, id):
        return self.truth_time[self.index_by_id(id)]

    def truth_pdgid_by_id(self, id):
        return self.truth_pdgid[self.index_by_id(id)]

    @property
    def energy(self):
        return self.x[:,0]

    @property
    def etahit(self):
        return self.x[:,1]

    @property
    def zerofeature(self):
        return self.x[:,2]

    @property
    def thetahit(self):
        return self.x[:,3]

    @property
    def rhit(self):
        return self.x[:,4]

    @property
    def xhit(self):
        return self.x[:,5]

    @property
    def yhit(self):
        return self.x[:,6]

    @property
    def zhit(self):
        return self.x[:,7]

    @property
    def time(self):
        return self.x[:,8]

    @property
    def select_em_hits(self):
        return np.isin(np.abs(self.truth_pdgid), np.array([11, 22, 111]))

    @property
    def select_mip_hits(self):
        return np.isin(np.abs(self.truth_pdgid), np.array([13]))

    @property
    def select_noise_hits(self):
        return self.y <= 0

    @property
    def select_signal_hits(self):
        return self.y > 0

    @property
    def select_had_hits(self):
        return (self.select_signal_hits & (~self.select_em_hits) & (~self.select_mip_hits))

    @property
    def em_energy_fraction(self):
        return self.energy[self.select_em_hits].sum() / self.energy[self.select_signal_hits].sum()

    @property
    def had_energy_fraction(self):
        return self.energy[self.select_had_hits].sum() / self.energy[self.select_signal_hits].sum()

    @property
    def mip_energy_fraction(self):
        return self.energy[self.select_mip_hits].sum() / self.energy[self.select_signal_hits].sum()


class TestYielderSingleTruthShower(TestYielder):
    def event_filter(self, event: Event):
        total_energy = event.energy[event.select_signal_hits].sum()
        for id in np.unique(event.y):
            if id == 0: continue
            shower_energy = event.energy[event.y==id].sum()
            if shower_energy / total_energy > .95:
                print(
                    f'{shower_energy=}, {total_energy=}, '
                    f'r={shower_energy/total_energy}, pdgid={event.truth_pdgid_by_id(id)}'
                    )
                return True
        return False


class Prediction:
    def __init__(self, pass_noise_filter, pred_betas, pred_cluster_space_coords):
        self.pass_noise_filter = pass_noise_filter
        self.pred_betas = pred_betas
        self.pred_cluster_space_coords = pred_cluster_space_coords


def cluster(prediction, tbeta=.2, td=.5):
    clustering_filtered = oc.get_clustering_np(
        prediction.pred_betas, prediction.pred_cluster_space_coords, tbeta=tbeta, td=td
        )+1
    clustering = np.zeros(prediction.pass_noise_filter.shape)
    clustering[np.flatnonzero(prediction.pass_noise_filter)] = clustering_filtered
    return clustering


def make_matches(event, prediction, tbeta=.2, td=.5, clustering=None):
    if clustering is None: clustering = cluster(prediction, tbeta, td)
    i1s, i2s, _ = match(event.y, clustering, weights=event.energy)
    matches = group_matching(i1s, i2s)
    return matches


def get_matched_vs_unmatched(event: Event , clustering, matches, noise_index=0):
    matched_truth = []
    matched_pred = []
    for truth_ids, pred_ids in matches:
        matched_truth.extend(truth_ids)
        matched_pred.extend(pred_ids)
    all_truth_ids = set(np.unique(event.y))
    all_pred_ids = set(np.unique(clustering))
    all_truth_ids.discard(noise_index)
    all_pred_ids.discard(noise_index)
    unmatched_truth = np.array(list(all_truth_ids - set(matched_truth)))
    unmatched_pred = np.array(list(all_pred_ids - set(matched_pred)))

    select_matched_truth = np.in1d(event.y, matched_truth)
    select_matched_pred = np.in1d(clustering, matched_pred)
    select_unmatched_truth = np.in1d(event.y, unmatched_truth)
    select_unmatched_pred = np.in1d(clustering, unmatched_pred)

    nhits = (event.y != noise_index).sum()
    total_truth_energy = event.energy[event.y != noise_index].sum()
    total_pred_energy = event.energy[clustering != noise_index].sum()

    stats = Stats()
    stats.add('n_showers_truth', len(all_truth_ids))
    stats.add('n_showers_pred', len(all_pred_ids))
    stats.add('n_showers_unmatched_truth', len(unmatched_truth))
    stats.add('n_showers_unmatched_pred', len(unmatched_pred))

    all_truth_pdgids = np.array([event.truth_pdgid_by_id(id) for id in all_truth_ids])
    unmatched_truth_pdgids = np.array([event.truth_pdgid_by_id(id) for id in unmatched_truth])

    count_type_fns = {
        'em' : lambda pdgids: np.in1d(np.abs(pdgids), np.array([11, 22, 111])).sum(),
        'mip' : lambda pdgids: np.in1d(np.abs(pdgids), np.array([13])).sum(),
        'had' : lambda pdgids: (~np.in1d(np.abs(pdgids), np.array([11, 22, 111, 13]))).sum(),
        }

    # print('allcat unmatched/total:', len(unmatched_truth), len(all_truth_ids))
    for i_cat, cat in enumerate(['em', 'had', 'mip']):
        count_type = count_type_fns[cat]
        n_truth_showers_this_cat = count_type(all_truth_pdgids)
        n_unmatched_truth_showers_this_cat = count_type(unmatched_truth_pdgids)
        if n_truth_showers_this_cat == 0: continue
        # print(cat, unmatched_truth_pdgids, n_unmatched_truth_showers_this_cat, n_truth_showers_this_cat)
        stats.add(f'n_showers_truth_{cat}', n_truth_showers_this_cat)
        stats.add(f'n_showers_unmatched_truth_{cat}', n_unmatched_truth_showers_this_cat)

    stats.add('nhits_matched_truth', select_matched_truth.sum())
    stats.add('nhits_matched_pred', select_matched_pred.sum())
    stats.add('nhits_unmatched_truth', select_unmatched_truth.sum())
    stats.add('nhits_unmatched_pred', select_unmatched_pred.sum())
    stats.add('hitenergy_matched_truth', event.energy[select_matched_truth].sum())
    stats.add('hitenergy_matched_pred', event.energy[select_matched_pred].sum())
    stats.add('hitenergy_unmatched_truth', event.energy[select_unmatched_truth].sum())
    stats.add('hitenergy_unmatched_pred', event.energy[select_unmatched_pred].sum())
    stats.add('fraction_nhits_matched_truth', select_matched_truth.sum()/nhits)
    stats.add('fraction_nhits_matched_pred', select_matched_pred.sum()/nhits)
    stats.add('fraction_nhits_unmatched_truth', select_unmatched_truth.sum()/nhits)
    stats.add('fraction_nhits_unmatched_pred', select_unmatched_pred.sum()/nhits)
    stats.add('fraction_hitenergy_matched_truth', event.energy[select_matched_truth].sum()/total_truth_energy)
    stats.add('fraction_hitenergy_matched_pred', event.energy[select_matched_pred].sum()/total_pred_energy)
    stats.add('fraction_hitenergy_unmatched_truth', event.energy[select_unmatched_truth].sum()/total_truth_energy)
    stats.add('fraction_hitenergy_unmatched_pred', event.energy[select_unmatched_pred].sum()/total_pred_energy)
    return stats


def is_np_array(thing):
    return hasattr(thing, 'shape') # Seems more reliable than checking against np.array

class Stats:
    """Container for statistics per object (event or match)
    
    Useful to keep track of quantities per object, and then combine
    for many objects.
    """
    def __init__(self):
        self.d = {}

    def __getitem__(self, key):
        return self.d[key]

    def add(self, key, val):
        """Add single value to a key"""
        val = np.expand_dims(np.array(val), 0)
        if key not in self.d:
            self.d[key] = val
        else:
            self.d[key] = np.concatenate((self.d[key], val))

    def extend(self, other):
        """Extend with another Stats object"""
        for k, v in other.d.items():
            if k in self.d:
                self.d[k] = np.concatenate((self.d[k], v))
            else:
                self.d[k] = v

    def __len__(self):
        for key, val in self.d.items():
            return len(val)

def dump_stats(outfile, stats):
    outfile = _make_parent_dirs_and_format(outfile)
    np.savez(outfile, **stats.d)

def load_stats(outfile):
    stats = Stats()
    stats.d = dict(np.load(outfile))
    return stats

def get_category(truth_ids):
    truth_ids = np.abs(truth_ids)
    exclusively_these_ids = lambda test_ids: np.all(np.in1d(truth_ids, np.array(test_ids)))
    any_of_these_ids = lambda test_ids: np.any(np.in1d(truth_ids, np.array(test_ids)))
    if exclusively_these_ids([11, 22, 111]):
        cat = 0 # EM
    elif exclusively_these_ids([13]):
        cat = 2 # MIP
    elif any_of_these_ids([11, 22, 111, 13]):
        # If any particle is EM or MIP, but not *all*, the category must be mixed
        cat = 3 # MIX
    else:
        # The truth ids must be exclusively hadronic
        cat = 1 # HAD
    return cat


def signal_to_noise_confusion_matrix(event, clustering, norm=False):
    # Turn all signal (index > 0) into simply True:
    yp = clustering.astype(bool)
    yt = event.y.astype(bool)
    confmat = np.array([
        [((yt == 0) & (yp == 0)).sum(), ((yt == 1) & (yp == 0)).sum()],
        [((yt == 0) & (yp == 1)).sum(), ((yt == 1) & (yp == 1)).sum()]
        ])
    confmat_hitenergy = np.array([
        [event.energy[(yt == 0) & (yp == 0)].sum(), event.energy[(yt == 1) & (yp == 0)].sum()],
        [event.energy[(yt == 0) & (yp == 1)].sum(), event.energy[(yt == 1) & (yp == 1)].sum()]
        ])
    if norm:
        confmat = confmat / confmat.sum() # Cannot do in place unless casting
        confmat_hitenergy /= confmat_hitenergy.sum()
    return np.stack((confmat, confmat_hitenergy))



def ids_to_selection(ids, clustering):
    np.isin(clustering, ids)


def statistics_per_match(event: Event, clustering, matches):
    stats = Stats()

    for truth_ids, pred_ids in matches:
        ebound_truth = 0.
        sel_truth_hits = np.zeros_like(event.y, dtype=bool)
        for truth_id in truth_ids:
            sel = event.y==truth_id
            index = sel.argmax()
            sel_truth_hits[sel] = True
            ebound_truth += event.truth_e_bound[index]

        stats.add('ebound_truth', ebound_truth)
        stats.add('eta_truth', np.average(event.etahit[sel_truth_hits], weights=event.energy[sel_truth_hits]))

        sel_pred_hits = np.zeros_like(event.y, dtype=bool)
        for pred_id in pred_ids: sel_pred_hits[clustering==pred_id] = True

        stats.add(
            'energy_iou',
            event.energy[sel_truth_hits & sel_pred_hits].sum() / event.energy[sel_truth_hits | sel_pred_hits].sum()
            )
        stats.add('category', get_category(np.unique(event.truth_pdgid[sel_truth_hits])))
        print('category', get_category('category', np.unique(event.truth_pdgid[sel_truth_hits])))
        stats.add('nhits_pred', sel_pred_hits.sum())
        stats.add('esum_pred', event.energy[sel_pred_hits].sum())
        stats.add('nhits_truth', sel_truth_hits.sum())
        stats.add('esum_truth', event.energy[sel_truth_hits].sum())
        stats.add('n_pred', len(pred_ids))
        stats.add('n_truth', len(pred_ids))

    return stats


def base_colorwheel():
    colorwheel = ColorWheel()
    colorwheel.assign(-1, '#bfbfbf')
    colorwheel.assign(0, '#bfbfbf')
    return colorwheel


# ____________________________________________-
# Plotly stuff

def cube_pdata(xmin, xmax, ymin, ymax, zmin, zmax):
    import plotly.graph_objects as go
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    unit_cube_edges = np.array([
        [[0,0,0], [1,0,0]],
        [[1,0,0], [1,1,0]],
        [[0,1,0], [1,1,0]],
        [[0,0,0], [0,1,0]],
        # 
        [[0,0,1], [1,0,1]],
        [[1,0,1], [1,1,1]],
        [[0,1,1], [1,1,1]],
        [[0,0,1], [0,1,1]],
        #
        [[0,0,0], [0,0,1]],
        [[1,0,0], [1,0,1]],
        [[0,1,0], [0,1,1]],
        [[1,1,0], [1,1,1]],
        ])
    offset = np.array([[xmin, ymin, zmin], [xmin, ymin, zmin]])
    scale = np.array([[dx, dy, dz], [dx, dy, dz]])
    pdata = []
    for unit_edge in unit_cube_edges:
        edge = offset + unit_edge*scale
        pdata.append(go.Scatter3d(
            x=edge[:,2], y=edge[:,0], z=edge[:,1],
            # text=['a', 'b'],
            # textposition='bottom left',
            mode='lines+markers+text',
            marker=dict(size=0),
            line=dict(
                color='black',
                width=6
                ),
            ))
    return pdata

def compile_plotly_data(
    event: Event, clustering: np.array=None, colorwheel=None
    ):
    import plotly.graph_objects as go
    if colorwheel is None: colorwheel = base_colorwheel()
    if clustering is None: clustering = event.y

    pdata = []
    energy_scale = 20./np.average(event.energy)

    for cluster_index in np.unique(clustering):
        is_noise = cluster_index == 0
        size_scale = .7 if is_noise else 1.
        sel_cluster = (clustering == cluster_index)

        if isinstance(colorwheel, ColorwheelWithProps):
            if not cluster_index in colorwheel:
                colorwheel.assign(cluster_index, alpha=.6)
            props = colorwheel(cluster_index)
            color = props['color']
            opacity = props['alpha']
        else:
            color = colorwheel(cluster_index)
            opacity = .6 if is_noise else 1.

        pdata.append(go.Scatter3d(
            x = event.zhit[sel_cluster], y=event.xhit[sel_cluster], z=event.yhit[sel_cluster],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=size_scale*np.maximum(0., np.minimum(3., np.log(energy_scale*event.energy[sel_cluster]))),
                color=color,
                ),
            text=[f'e={e:.3f}<br>t={t:.3f}' for e, t in zip(event.energy[sel_cluster], event.time[sel_cluster])],
            hovertemplate=(
                f'x=%{{y:0.2f}}<br>y=%{{z:0.2f}}<br>z=%{{x:0.2f}}'
                f'<br>%{{text}}'
                f'<br>clusterindex={cluster_index}'
                f'<br>pdgid={int(event.truth_pdgid_by_id(cluster_index))}'
                f'<br>E_bound={event.truth_e_bound_by_id(cluster_index):.3f}'
                f'<br>sum(E_hit)={event.energy[sel_cluster].sum():.3f}'
                f'<br>'
                ),
            name = f'cluster_{cluster_index}',
            opacity=opacity
            ))
    # pdata.extend(cube_pdata(
    #     min(event.xhit), max(event.xhit),
    #     min(event.yhit), max(event.yhit),
    #     min(event.zhit), max(event.zhit)
    #     ))
    return pdata

def compile_plotly_data_clusterspace(
    event: Event, prediction: Prediction, clustering: np.array=None, colorwheel=None
    ):
    import plotly.graph_objects as go
    if colorwheel is None: colorwheel = base_colorwheel()
    if clustering is None: clustering = event.y
    clustering = clustering[prediction.pass_noise_filter]

    pdata = []
    print(prediction.pred_cluster_space_coords.shape)
    coords = pca_down(prediction.pred_cluster_space_coords)
    print(coords.shape)
    assert coords.shape == (clustering.shape[0], 3)

    for cluster_index in np.unique(clustering):
        sel_cluster = (clustering == cluster_index)
        pdata.append(go.Scatter3d(
            x = coords[sel_cluster,0], y=coords[sel_cluster,1], z=coords[sel_cluster,2],
            mode='markers', 
            marker=dict(
                line=dict(width=0),
                size=1.,
                color=colorwheel(int(cluster_index)),
                ),
            hovertemplate=(
                f'x=%{{x:0.2f}}<br>y=%{{y:0.2f}}<br>z=%{{z:0.2f}}'
                ),
            name = f'cluster_{cluster_index}'
            ))
    return pdata


def _make_parent_dirs_and_format(outfile, touch=False):
    import os, os.path as osp
    from time import strftime
    outfile = strftime(outfile)
    outdir = osp.dirname(osp.abspath(outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    if touch:
        with open(outfile, 'w'):
            pass
    return outfile

def single_pdata_to_file(
    outfile, pdata, mode='w', title=None, width=600, height=None, include_plotlyjs='cdn'
    ):
    import plotly.graph_objects as go
    scene = dict(xaxis_title='z (cm)', yaxis_title='x (cm)', zaxis_title='y (cm)', aspectmode='cube')
    if height is None: height = width
    fig = go.Figure(data=pdata, **(dict(layout_title_text=title) if title else {}))
    fig.update_layout(width=width, height=height, scene=scene)
    fig_html = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)

    print('Writing to', outfile)

#     import re
#     mode_bar_plugin = """\"modeBarButtons\": [[{
#     name: "save camera",
#     click: function(gd) {
#       var scene = gd._fullLayout.scene._scene;
#       scene.saveCamera(gd.layout);     
#     }
#   }, 
#     "toImage"
#   ]],"""
#     print(re.search(r'\}\],\s*\{', fig_html))
#     fig_html = re.sub(r'\}\],\s*\{', '}]\n{' + mode_bar_plugin + '\n', fig_html)

    outfile = _make_parent_dirs_and_format(outfile)
    with open(outfile, mode) as f:
        f.write(fig_html)

def side_by_side_pdata_to_file(
    outfile, pdata1, pdata2,
    title1=None, title2=None, width=600, height=None, include_plotlyjs='cdn',
    mode='w', legend=True
    ):
    import plotly.graph_objects as go
    scene = dict(xaxis_title='z (cm)', yaxis_title='x (cm)', zaxis_title='y (cm)', aspectmode='cube', xaxis_mirror=True)
    if height is None: height = width
    fig1 = go.Figure(data=pdata1, **(dict(layout_title_text=title1) if title1 else {}))
    fig1.update_layout(width=width, height=height, scene=scene, showlegend=legend)
    fig2 = go.Figure(data=pdata2, **(dict(layout_title_text=title2) if title2 else {}))
    fig2.update_layout(width=width, height=height, scene=scene, showlegend=legend)
    fig1_html = fig1.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    fig2_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    divid1 = fig1_html.split('<div id="',1)[1].split('"',1)[0]
    divid2 = fig2_html.split('<div id="',1)[1].split('"',1)[0]
    id1 = str(uuid.uuid4())[:6]
    id2 = str(uuid.uuid4())[:6]
    # Compile html: Sync camera angles in javascript
    html = (
        f'<div style="width: 47%; display: inline-block">\n{fig1_html}\n</div>'
        f'\n<div style="width: 47%; display: inline-block">\n{fig2_html}\n</div>'
        f'\n<script>'
        f'\nvar graphdiv_{id1} = document.getElementById("{divid1}");'
        f'\nvar graphdiv_{id2} = document.getElementById("{divid2}");'
        f'\nvar isUnderRelayout_{id1} = false'
        f'\ngraphdiv_{id1}.on("plotly_relayout", () => {{'
        f'\n    // console.log("relayout", isUnderRelayout_{id1})'
        f'\n    if (!isUnderRelayout_{id1}) {{'
        f'\n        Plotly.relayout(graphdiv_{id2}, {{"scene.camera": graphdiv_{id1}.layout.scene.camera}})'
        f'\n        .then(() => {{ isUnderRelayout_{id1} = false }}  )'
        f'\n        }}'
        f'\n    isUnderRelayout_{id1} = true;'
        f'\n    }})'
        f'\nvar isUnderRelayout_{id2} = false'
        f'\ngraphdiv_{id2}.on("plotly_relayout", () => {{'
        f'\n    // console.log("relayout", isUnderRelayout_{id2})'
        f'\n    if (!isUnderRelayout_{id2}) {{'
        f'\n        Plotly.relayout(graphdiv_{id1}, {{"scene.camera": graphdiv_{id2}.layout.scene.camera}})'
        f'\n        .then(() => {{ isUnderRelayout_{id2} = false }}  )'
        f'\n        }}'
        f'\n    isUnderRelayout_{id2} = true;'
        f'\n    }})'
        f'\n</script>'
        )
    outfile = _make_parent_dirs_and_format(outfile)
    with open(outfile, mode) as f:
        f.write(html)
