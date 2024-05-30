# Fork Note

This is a fork of https://github.com/tklijnsma/hgcal_training_scripts/. The purpose of this repo is to do an energy resolution analysis of the GNN algorithm that reconstructs particles at the CMS HGCal. The actual energy resolution analysis is at energy_resolution analysis.py. The rest of the files are the training and plotting scripts done by THomas Klijnsma in the original repo.

# Note

This is a repo with some basic training and plotting scripts. This documentation is meant to get started but is in no way complete.

# Setup

```
conda create -n hgcal-ml python=3.9
conda activate hgcal-ml

# For GPU support. Cuda installations differ somewhat between platforms.
# CPU version of pytorch works too.
conda install pytorch cudatoolkit=11.1 -c pytorch -c nvidia

conda install pytorch-geometric -c rusty1s -c conda-forge

git clone https://github.com/cms-pepr/pytorch_cmspepr.git
pip install -e pytorch_cmspepr

git clone https://github.com/tklijnsma/hgcal_training_scripts.git
cd hgcal_training_scripts
```

Some more packages might be needed, e.g. `matplotlib`, `plotly`, `scikit-learn`, etc. depending on what you want to do.

# Get input

```
xrdcp root://cmseos.fnal.gov//store/user/klijnsma/hgcal/taus_2021.tar.gz .
tar xvf taus_2021.tar.gz
```

Also get a checkpoint:

```
xrdcp root://cmseos.fnal.gov//store/user/klijnsma/hgcal/ckpts/ckpt_train_taus_integrated_noise_Oct20_212115_best_397.pth.tar .
```

# Create event displays & histograms

Use the `plots` or `stats` functions from [plots_to_files_integrated_noise_filter.py](plots_to_files_integrated_noise_filter.py). Some basic plotting code is in [plot_histograms.ipynb](plot_histograms.ipynb).
