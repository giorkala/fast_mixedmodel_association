"""
Create the necessary HDF5 files, according to partition and subsets.
Invoke as
`python create_hdf5_py annotation pheno_prefix hdf_prefix N_comp`
"""
from fma import find_chromosomes, find_components, convert_bed_to_hdf5 
import pandas as pd
from pysnptools.snpreader import Bed
import h5py, sys, os
from tqdm import tqdm
import numpy as np

work_dir = "FIX_THIS"

## 1. load necessary files ##
geno="/FIX_THIS/plink_maf0.01/ukb_app43206_500k.bgl5-phased.maf0.01"
snp_on_disk = Bed(geno, count_A1=True)
chr_map = find_chromosomes(geno)
block_ID_all = find_components( work_dir+sys.argv[1], int(sys.argv[4]), chr_map )
Traits = pd.read_csv("FIX_THIS/results_ukbb_446k/phenotypes.tab", sep='\s+')
Traits = Traits.set_index( Traits["FID"])

# check if any individual is less than 50% phenotyped
phen_thres=0.50
N_phen = Traits.shape[1] - 2 # exclude FID, IID
samples_with_missing = np.where( Traits.isna().sum(axis=1)/N_phen > phen_thres )[0]
print("{0} samples are less than {1}% phenotyped and will be excluded.".format(len(samples_with_missing),100*phen_thres))
Traits.drop( Traits.FID.iloc[samples_with_missing], axis=0, inplace=True) 
Traits = Traits.fillna(Traits.mean()) # impute the missing values with the mean of each trait
assert sum(Traits.isna().sum())==0, "Some phenotypes stll have missing values!"
print("Samples with available genotypes and phenotypes to keep for analysis:", len(Traits) )

## 2. create the sets of samples in the {0,...,N} index ##
index = pd.DataFrame()
index[0] = range(len(snp_on_disk.iid[:,0]))
index[1] = pd.to_numeric(snp_on_disk.iid[:,0])
index.set_index( index[1], inplace=True ) # map from UKBB-id to {0,...,N} id
index = index.loc[Traits.index] # keep only the highly-phenotyped individuals
# set(index.index).intersection(Traits.index)
subsets = []
for i in range(1,6):
    df = pd.read_csv(work_dir+"{0}{1}.tab".format(sys.argv[2], i), sep='\t', usecols=["FID"])
    subsets.append( np.sort(index.loc[ set(index.index).intersection(df.FID) ][0]) )
    
## 3. initialise the hdf5 files ##
datasets = {}
M = sum([ len(x) for x in block_ID_all.values()])
for i in range(1,6):
    save_as = work_dir+"{0}{1}.hdf5".format(sys.argv[3],i)
    if os.path.exists(save_as): os.remove(save_as)
    datasets[i] = [h5py.File( save_as, 'w'), 0]
    datasets[i][1] = datasets[i][0].create_dataset('genotypes', (M, len(subsets[i-1])), chunks=(500, len(subsets[i-1])), dtype= 'int8')
    print("Will create an hdf5 file for N={0}, based on {1}, named as {2}".format(len(subsets[i-1]), sys.argv[1], save_as))

## 3. start writing on the hdf5 files ##
left=0; right=0
for b in tqdm(block_ID_all):
    right += len(block_ID_all[b])
    G = snp_on_disk[ :, block_ID_all[b] ].read( dtype='int8' ,  _require_float32_64=False ).val
#     print("check", end=', ')
    for i in range(1,6):
        datasets[i][1][left:right, :] = G[ subsets[i-1], :].T
    left += len(block_ID_all[b])
    
for i in range(1,6):
    datasets[i][0].close()
# end-of-file