#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A stand-alone wrapper to use FMA-MC-MT directly from a terminal.
This deals with different sets of phenotyped and genotyped samples and works only for the intersection.
The same for covariates. Maybe such variables could be mean-imputed?
We'll assume that the files with covariates and phenotype can be different, and that all the given covariates will be used.
TODO: How to deal with missing phenotypes?
Created on Mon May 31 16:03:23 2021
@author: kalantzi
"""
#%% setup and flags
import argparse, time, os, sys
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description="Perform Randomized Complex Trait Aanalysis for a multiple genetic components. \nCan be invoked as \n \t`python invoke_mt.py -g path_to_plink -p path_to_pheno --VC 0.25,0.75 -o where_to_save`", formatter_class=RawTextHelpFormatter)
parser.add_argument("--bfile", "-g", help="prefix for bed/bim/fam files", type=str)
parser.add_argument("--phenoFile", "-p", help="phenotype file; should be in \"FID,IID,Trait\" format and tsv", type=str)
parser.add_argument("--covarFile", "-c", help="file with covariates; should be in \"FID,IID,Var1,Var2,...\" format and tsv", type=str)
parser.add_argument("--output", "-o", help="prefix for where to save any results or files")
parser.add_argument("--PoolSize", help="number of batches to process in parallel", default=5, type=int)
parser.add_argument("--nThreads", help="number of computational threads to be used", default='1', type=str)
parser.add_argument("--nCalibr", help="number of SNPs for calibration", default=3, type=int)
parser.add_argument("--maxIters", help="max number of CG iterations", default=50, type=int)
parser.add_argument("--vcThres", help="min threshold for which VC to consider", default=0.0005, type=float)
parser.add_argument("--annot","-a", help="file with annotation; one column per component; no overlapping", default='1', type=str)
group = parser.add_mutually_exclusive_group()
group.add_argument("--rhe", help="run RHEmc on the specified geno-pheno", action="store_true")
group.add_argument("--rheReadLog", help="don't run RHEmc but read from a file", type=str)
parser.add_argument("--useHDF", help="create a new hdf5 with a given filename", default=None)
parser.add_argument("--inMem", help="load all the genotypes in memory - in dev", default=False, action="store_true" )
parser.add_argument("--faster", help="speed-up using pseudo-loco", default=False, action="store_true" )
parser.add_argument("--SNPtest", help="get test statistics on all SNPs", action="store_true")
parser.add_argument("--debug", help="print intermediate info", action="store_true")
#parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], help="increase output verbosity")

args = parser.parse_args()
# multi-threading: this should be kept low as more processes are more efficient and each process uses `nThreads`
# e.g. 6 process with 1 thread each is probably better than 3x2, and way more efficient than 2x3 or 1x6.
os.environ['MKL_NUM_THREADS'] = args.nThreads; os.environ['OPENBLAS_NUM_THREADS'] = args.nThreads
os.environ["OPENBLAS_NUM_THREADS"] = args.nThreads

import numpy as np
from pysnptools.snpreader import Bed
import pandas as pd
from fma import RunRHE, PreparePhenoRHE, vc_adjustment 
from fma import FMA_mem_version, FMA_streaming_mt, FMA_test_many
rhemc="/FIXTHIS/RHEmc"
rhemc="/software/team281/gk18/multi-trait/build/MTMA_VC"

class Logger(object):
    # see https://stackoverflow.com/questions/11325019/
    # or  https://stackoverflow.com/questions/14906764/
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
            # TODO; it still doesnt print everything immidiately (I think)
    def flush(self) :
        for f in self.files:
            f.flush()

logfile = open(args.output+'.log', 'w')
sys.stdout = Logger(sys.stdout, logfile)
durations={}

if __name__ == "__main__":

    time_start = time.perf_counter()
    
    if args.debug:
        debug=True
        print("FMA - run with options:")
        print(', '.join(f'{k}={v}' for k, v in vars(args).items()))
    else:
        debug=False
        
    if os.path.exists( args.bfile+".bed" ):
        print("Genotypes will be loaded from", args.bfile+".bed")
    else:
        print("ERROR: missing genotype files.")
    snp_on_disk = Bed(args.bfile, count_A1=True)
    samples_geno = [int(x) for x in snp_on_disk.iid[:,0] ]
        
    #%% Phenotype loading and alignment
    Traits = pd.read_csv( args.phenoFile, sep='\s+', low_memory=False)
    Traits = Traits.set_index( Traits["FID"])
    Traits = Traits.dropna(subset=[Traits.columns[0]])
    N_phen = Traits.shape[1] -2 # exclude FID, IID
    print("{0} phenotypes were loaded for {1} samples.".format(N_phen, Traits.shape[0]))
    # remove those without genotypes
    Traits = Traits.drop( set(Traits['FID']).difference(set(samples_geno).intersection(Traits.FID)), axis=0) 
    Traits.reindex(sorted(Traits.columns), axis=1) # ensure proper alignment with the VC estimates
    
    # check if any individual is less than 50% phenotyped
    phen_thres=0.50
    samples_with_missing = np.where( Traits.isna().sum(axis=1)/N_phen > phen_thres )[0]
    print("{0} samples are less than {1}% phenotyped and will be excluded.".format(len(samples_with_missing),100*phen_thres))
    Traits.drop( Traits.FID.iloc[samples_with_missing], axis=0, inplace=True) 
    
    if debug:   
        print("Missingness ratios for current sample (these will be mean-imputed):") # exclude FID/IID
        for trait in Traits.columns[2:]:
            print("{0:.2f}%, {1}".format(Traits[trait].isnull().sum()/len(Traits)*100, trait))
    Traits = Traits.fillna(Traits.mean()) # impute the missing values with the mean of each trait
    assert sum(Traits.isna().sum())==0, "Some phenotypes stll have missing values!"
    
    samples = {}
    for i in range(len(samples_geno)): 
        if str(snp_on_disk.iid[i,0]) in Traits.index:
            samples[i] = str(snp_on_disk.iid[i,0])
    N_total = len(Traits)
    samples = list(samples.keys()) 
    if debug: print("Samples with available genotypes and phenotypes to keep for analysis:", N_total )
    
    #%% covariate adjustment
    if args.covarFile != None:
        print("Loading and preparing covariates...")
        df_covar = pd.read_csv(args.covarFile, sep='\s+', low_memory=False)
        df_covar = df_covar.set_index( df_covar["FID"])
        # keep only the intersection of pheno/geno/covar-typed samples
        samples_to_keep = set(samples_geno).intersection(Traits.FID).intersection(df_covar.FID)
        df_covar = df_covar.drop( set(df_covar['FID']).difference(samples_to_keep), axis=0) 
        df_covar = df_covar.fillna(df_covar.median())
        
        Traits = Traits.drop( set(Traits['FID']).difference(samples_to_keep), axis=0) 
        samples = {}
        for i in range(len(samples_geno)): 
            if str(snp_on_disk.iid[i,0]) in df_covar.index:
                samples[i] = str(snp_on_disk.iid[i,0])
        # now we need to align the individuals in the df for the math to follow
        df_covar = df_covar.reindex( samples.values() )
        Traits = Traits.reindex( samples.values() )
        N_total = len(Traits)
        print("Samples with available genotypes, phenotypes, and covariates to keep for analysis:", N_total )
        print("Covariates to use: " + " ".join(df_covar.columns[2:].values))        
        select = df_covar.columns[2:] # select all but FID, IID
        # select = ['Sex', 'Age', 'Site', "Smoking", 'PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10']
        W = np.concatenate([df_covar[select].to_numpy(), np.ones((N_total,1))], axis=1)
        samples = list(samples.keys()) 
        
        for T in range(N_phen):
            Trait = Traits.iloc[:,T+2]
            Trait -= W.dot( np.linalg.inv(W.T.dot(W)) ).dot( W.T.dot(Trait) )
            Trait -= np.mean(Trait); Trait /= np.std(Trait)
            Traits.iloc[:,T+2] = Trait
        print("The traits are now adjusted with respect to the covariates, mean-centered and of unit variance.")
        del df_covar, Trait
    else:
        print("\nWARNING: No covariates will be used! Are the traits already adjusted?")
        samples_to_keep = set(samples_geno).intersection(Traits.FID)
        Traits = Traits.drop( set(Traits['FID']).difference(samples_to_keep), axis=0) 
        # Traits = Traits.reindex( samples.values() )
        samples = {}
        for i in range(len(samples_geno)): 
            if str(snp_on_disk.iid[i,0]) in Traits.index:
                samples[i] = str(snp_on_disk.iid[i,0])
        N_total = len(Traits)
        samples = list(samples.keys()) 
        print("Samples with available genotypes and phenotypes to keep for analysis:", N_total )
        
    # we save a new file with the adjusted trait for RHEmc and for reference
    adj_pheno_file = ".".join( [args.output,"adjusted_traits","phen"])
    PreparePhenoRHE( Traits, args.bfile+".fam", adj_pheno_file )
    print("Adjusted phenotypes are saved in "+adj_pheno_file)
    
    Traits = Traits.drop( labels=["FID","IID"], axis=1)
    
    #%% annotation and variance-component estimates
    if args.annot == '1': 
        # this is the default; if no file is given, we create one now (but most likely it exists already)
        K = 1 
        print("Saving a file with a ''null'' annotation as "+args.output+".uniform.annot.")
        temp = pd.DataFrame(np.ones(snp_on_disk.shape[1], dtype=int))
        temp.to_csv(args.output+".uniform.annot",header=None,index=None,sep=' ')
        args.annot = args.output+".uniform.annot"
    else: # now we assume that a file is given
        assert os.path.isfile(args.annot), "Annotation file doesn't exist!"
        temp = pd.read_csv(args.annot, sep=' ', header=None, nrows=1)
        K = temp.shape[1] # infer the number of components from the annotation
    
    if type(args.rheReadLog)==str:
        print("Read RHEmc's estimates from: "+args.rheReadLog)
        assert os.path.isfile(args.rheReadLog), "Logfile "+args.rheReadLog+" doesn't exist!"
        VC = pd.read_csv(args.rheReadLog, sep='\s+', header=None)
        VC.set_index(VC[0], inplace=True)
        VC.drop(columns=0, axis=1, inplace=True)
    else:
        print("Running RHEmc to estimate variances:")
        tic = time.perf_counter()
        VC = RunRHE( args.bfile, adj_pheno_file, args.annot, args.output+".rhe.log", rhemc ) #sigma_g_hat, sigma_e_hat
        durations['rhe-mc'] = time.perf_counter()-tic
        print("Done. Duration =", durations['rhe-mc'])
        # TODO: transform the new files to a readable format.
        print("ERROR: Not yet finalised!")
    
    ## check if Traits and VC are aligned
    VC = VC.loc[Traits.columns,:K+1]
    assert (VC.index == Traits.columns).all(), "Traits and VC estimates are not properly aligned!"
    if debug: print("Variance components to be used", VC)

    if args.vcThres>0.0:
        VC.iloc[:,-1] = np.abs(VC.iloc[:,-1]) # fix bad Ve estimates
        VC[VC<0] = 0
        if debug: print(VC)
        print("Pruning components with variance <", args.vcThres)
        for T in range(N_phen):
            VC.iloc[T,:] = vc_adjustment(VC.iloc[T,:].to_numpy(), args.vcThres)

        keep = np.where( np.mean(VC) > args.vcThres/VC.shape[0] )[0] # this is the average across traits, so it has to be low in order to discard only the almost-zero components
        VC_new = VC.iloc[:,keep]
        if len(keep) < VC.shape[1]:
            # create a new annotation
            annot = pd.read_csv( args.annot, sep=' ', header=None )
            annot = annot.iloc[:,keep[:-1]-1] # exclude the Ve column and convert indices to {0,..,16}
            args.annot = args.output+".annot"
            annot.to_csv( args.annot, sep=' ', header=None, index=None)
        # adjust again to have normalised values
        for T in range(N_phen):
            VC_new[:].iloc[T] = vc_adjustment(VC_new.iloc[T,:].to_numpy(), 0.0)
        
        VC_new.to_csv( args.output+".rhe", sep='\t', header=None)
        VC = VC_new.to_numpy().T
    else:
        VC = VC.to_numpy().T

    if debug: print("Variance components to be used", VC)

    if K==1:
        arg_uniform = True
    else:
        arg_uniform = False
        
    # each case returns "[residuals_all, calibrators, q]", where residuals_all is a chrom-based dict
    if args.inMem:
        # NOTE: not optimised!
        fma_pack = FMA_mem_version ( args.bfile, Traits, samples, args.annot, VC, arg_uniform, arg_Ncal=args.nCalibr, cores=args.PoolSize, cg_max_iters=args.maxIters, arg_debug=debug )
    else:
        fma_pack = FMA_streaming_mt( args.bfile, Traits, samples, args.annot, VC, arg_uniform, arg_Ncal=args.nCalibr, cores=args.PoolSize, cg_max_iters=args.maxIters, arg_debug=debug, useHDF=args.useHDF )
    ## calculate the test statistics for all the array/model SNPs
    if args.SNPtest:
        FMA_test_many( args.bfile, fma_pack[0], fma_pack[1], samples, args.output, verbose=True )
    
    ## save the calibration factors
    df = pd.DataFrame.from_dict(fma_pack[1], orient='index')
    df.to_csv( args.output+".calibrators", header=None, sep='\t', float_format='%.6f')
        
    ## save the residuals, on a chrom-based order (more suitable for next step)
    Nchrom = len(fma_pack[0]) # deduce the number of chromosomes from the size of the dict
    for C in range(Nchrom):
        df = pd.DataFrame()
        df['FID'] = snp_on_disk.iid[ samples, 0] # we need FID and IID
        df['IID'] = snp_on_disk.iid[ samples, 1] # could use Traits.index only?
        for T,trait in enumerate(fma_pack[1].keys()): 
            df[trait] = fma_pack[0][C][:,T]
        df.to_csv( "{0}.loco{1}.residuals.gz".format( args.output,C+1), index=None, sep='\t', compression='gzip')

    print("\nResiduals are saved as {0}.loco*.residuals.gz".format( args.output ))
    
    print("\nTotal elapsed time for FMA = {0:.2f} seconds".format( time.perf_counter() - time_start )) #durations['rhe-mc'])
# end-of-file