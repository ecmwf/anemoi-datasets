import numpy as np
from anemoi.datasets import open_dataset
from anemoi.datasets.data.grids import Cutout

def test_cutout_initialization(lam_dataset_1, lam_dataset_2, global_dataset):
    """ Ensure that the Cutout class correctly initializes with multiple Limited 
    Area Models (LAMs) and a global dataset"""
    cutout = Cutout(
        [lam_dataset_1, lam_dataset_2, global_dataset], 
        axis=3,
        #adjust="all",
        )
    
    assert len(cutout.lams) == 2
    assert cutout.globe is not None
    assert len(cutout.masks) == 2

def test_cutout_mask_generation(lam_dataset, global_dataset):
    """" Ensure that the cutout_mask function correctly generates masks for LAMs 
    and excludes overlapping regions."""
    cutout = Cutout(
        [lam_dataset, global_dataset], axis=3)
    mask = cutout.masks[0]
    lam = cutout.lams[0]
    
    assert mask is not None
    assert isinstance(mask, np.ndarray)
    assert isinstance(cutout.global_mask, np.ndarray)
    assert mask.shape[-1] == lam.shape[-1]
    assert cutout.global_mask.shape[-1] == global_dataset.shape[-1]
    
    
def test_cutout_getitem(lam_dataset, global_dataset):
    """ Verify that the __getitem__ method correctly returns the appropriate 
    data when indexing the Cutout object """
    cutout = Cutout([lam_dataset, global_dataset], axis=3)
    
    data = cutout[0, :, :, :]
    expected_shape = cutout.shape[1:]
    assert data is not None
    assert data.shape == expected_shape
    
def test_latitudes_longitudes_concatenation(lam_dataset_1, lam_dataset_2, global_dataset):
    """ Ensure that latitudes and longitudes are correctly 
    concatenated from all LAMs and the masked global dataset."""
    cutout = Cutout(
        [lam_dataset_1, lam_dataset_2, global_dataset], 
        axis=3
        )
    
    latitudes = cutout.latitudes
    longitudes = cutout.longitudes
    
    assert latitudes is not None
    assert longitudes is not None
    assert len(latitudes) == cutout.shape[-1]
    assert len(longitudes) == cutout.shape[-1]
    
def test_overlapping_lams(lam_dataset_1, lam_dataset_2, global_dataset):
    """ Confirm that overlapping regions between LAMs and the global dataset are 
    correctly handled by the masks."""
    # lam_dataset_2 has to overlap with lam_dataset_1
    cutout = Cutout(
        [lam_dataset_1, lam_dataset_2, global_dataset], 
        axis=3
        )
    
    # Verify that the overlapping region in lam_dataset_2 is excluded
    assert np.count_nonzero(cutout.masks[1] == False) > 0
    
def test_open_dataset_cutout(lam_dataset_1, global_dataset):
    """ Ensure that open_dataset(cutout=[...]) works correctly with the new 
    Cutout implementation"""
    ds = open_dataset(
        cutout=[lam_dataset_1, global_dataset]
        )

    assert isinstance(ds, Cutout)
    assert len(ds.lams) == 1
    assert ds.globe is not None
    
    
def test_large_cutout_performance(large_lam_dataset, large_global_dataset):
    """  Visually confirm that the masked regions are correct"""
    cutout = Cutout(
        [large_lam_dataset, large_global_dataset], 
        axis=3
        )
    
    data = cutout[0, :, :, :]
    assert data is not None