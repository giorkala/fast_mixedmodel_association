#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A stand-alone wrapper for testing with FMA on BGEN.
This deals with different sets of phenotyped and genotyped samples and works only for the intersection.

Created on Sat Oct 15 17:22:34 2022

@author: kalantzi
"""
#%% Main functions
import argparse, time, os
from argparse import RawTextHelpFormatter
import numpy as np
import pandas as pd
from scipy.stats import chi2
from pysnptools.distreader import Bgen
from pysnptools.snpreader import Bed
from fma import FMA_test_many, find_chromosomes
# from multiprocessing.pool import Pool

def get_chi2_many( X, Y, W, K, minmaf=1e-5 ):
    """
    Author: Yiorgos + Hrushi
    DIY Linear regression with covariates and low memory footprint.
    X should be MxN, for sufficiently large M (e.g. one chromosome)
    Y is multi-phenotype but should also be mean centered.
    W needs to be a NxC array with covariates including ones (the constant).
    K  is the inverse of W.T.dot(W) (which we should calculate only once)
    minmaf is a filter 
    TODO: it would be faster if X was "muts x samples" ?
    """
    ## Preprocess genotypes first
    afreq = np.zeros(X.shape[1])
    nans = 0
    for snp in np.arange(X.shape[1]):
        isnan_at_snp = np.isnan(X[:, snp])
        freq = np.nansum(X[:, snp]) / np.sum(~isnan_at_snp)
        X[:, snp][isnan_at_snp] = 0
        # X[:, snp][~isnan_at_snp] -= freq # I'm not sure we need this!
        afreq[snp] = freq / 2
        if np.sum(isnan_at_snp)>0: nans += 1
        
    print("NaNs found:",nans)
    afreq = np.array([x if x<0.50 else 1-x for x in afreq])
    X = X[:, afreq > minmaf ]

    N, M = X.shape
    beta, chisq = np.zeros((Y.shape[1], M)), np.zeros((Y.shape[1], M))
    # temp = W.T.dot(X)
    temp = X.T.dot(W)
    var_X = np.array(
        [ X[:, v].dot(X[:, v]) - temp[v].T.dot( K.dot(temp[v]) ) for v in range(M)]
    ) 
    
    y_hat = Y - W.dot(K.dot(W.T.dot(Y))) # this is Py
    numerators = X.T.dot(y_hat) # this is xPy
    for pheno in np.arange(Y.shape[1]):
        var_y = Y[:, pheno].dot(y_hat[:, pheno])
        beta[pheno] = numerators[:, pheno] / var_X
        chisq[pheno] = ( N - W.shape[1] - 1 ) / (var_y * var_X / numerators[:, pheno] ** 2 - 1)
        # chisq[pheno] = N * numerators[:, pheno] ** 2 / var_y / var_X
        
    return beta, chisq, afreq

def FMA_test_chrom( C, snp_on_disk, residuals_wrap, calibrators, W, samples=None, min_maf=1e-5, batch_size=1000, saveas="FMA.sumstats", verbose=True ):
    """
    Test for association given the LOCO residuals and calibration factors, for many traits, but one chromosome.
    Can work both for BGEN and BED!
    The input could be just paths to files or arrays etc. It only saves the sumstats on disk. 
    There shouldnt be any problems with trait indexing, as long as the residuals and calibrators are aligned.
    INPUT:
        C: which chromosome to test for
        bgen: prefix for the bgen file, OR
        bfile: prexix for bed/bim/fam
        residuals: array form NxN_pheno 
        calibrators: dict of the form "phenoname" -> "calibr-factor"
        samples: index of individuals to include in analysis; 
    TODO: samples should be different for each trait, according to missingness
    """
    header = True # for the csv files
    N = len(samples)
    pheno_names = list(calibrators.keys())
    N_phen=len(calibrators)
    M_tested = 0; M_all = 0

    if verbose: 
        chisqr_all = np.zeros((snp_on_disk.shape[1],N_phen))*np.nan # just in order to print the average/median
        print("The verbose option is ON; this might use a lot of memory!")

    assert residuals.shape[0] == N, "Shape of residuals is not as expected!"
    assert residuals.shape[1] == N_phen, "Shape of residuals is not as expected!"

    for T in range(N_phen):
        if os.path.exists(".".join( [saveas,pheno_names[T],"sumstats.gz"])):
            header = False # we assume this is written earlier
    
    N_batches = snp_on_disk.sid_count//batch_size; # TODO: make this an option?
        
    K = np.linalg.inv(W.T.dot(W))
    
    for i in range(0, N_batches+1):
        index = np.arange(snp_on_disk.sid_count)[i*batch_size : min((i+1)*batch_size,snp_on_disk.sid_count)]
        geno_matrix = snp_on_disk[ samples, index ].read(  _require_float32_64=False ).val #dtype='int8' 
        col_nans = len(samples) - np.sum(np.isnan(geno_matrix),axis=0)
        
        # the next returns "Npheno x M"        
        beta, chisq, afreq = get_chi2_many( geno_matrix, residuals, W, K, min_maf )      
        mask_snps = afreq > min_maf
        M_tested += len(beta[0])
        M_all += len(afreq)
        for T in range(N_phen):
                
            sum_stats = pd.DataFrame(columns=["SNP", "CHR","BP","AFREQ","N","BETA","SE","CHISQ","P"]); #"CHISQ_LR",
            # sum_stats['SNP'] = [ x[1:] for x in snp_on_disk[ :, index ].sid[mask_snps] ] # snp_on_disk[ :,index ].sid
            sum_stats['SNP'] = snp_on_disk[ :, index ].sid[mask_snps] 
            sum_stats['CHR'] = [ x.split(":")[0] for x in snp_on_disk.sid[mask_snps] ]
            sum_stats['BP'] = np.array(snp_on_disk[:, index ].pos[:, 2], dtype="int")[mask_snps]
            sum_stats["AFREQ"] = afreq[mask_snps]
            # sum_stats['A1'] = temp[4][index]; sum_stats['A2'] = temp[5][index];
            sum_stats['N'] = col_nans[mask_snps]
            # the next is norm(x)^2 * norm(Vinv_y)^2:
            sum_stats["CHISQ"] = chisq[T] / calibrators[pheno_names[T]]
            sum_stats['BETA'] = beta[T] / calibrators[pheno_names[T]]
            sum_stats['SE'] = np.abs(sum_stats['BETA']) / np.sqrt(sum_stats["CHISQ"])
            sum_stats['P'] = chi2.sf( sum_stats['CHISQ'] , df=1)
            #  write on disk asynchronously; we write the header only the first time
            sum_stats.to_csv( ".".join( [saveas,pheno_names[T],"sumstats.gz"]), sep='\t', index=None, header=header, na_rep=np.nan, compression='gzip', mode='a') 
            if verbose: chisqr_all[index[afreq > min_maf],T] = sum_stats["CHISQ"]
        header = False # dont write the header again
            
    # print("Duration for test-statistics = {0:.2f} mins".format( (time.perf_counter()-time0)/60 ))
    print("SNPs tested = {0} and discarded due to MAF = {1}".format( M_tested, M_all - M_tested ))
    if verbose: 
        for T in range(N_phen):
            print(pheno_names[T])
            print("Mean FMA  : {0:.4f}  ({1} good SNPs)   lambdaGC: {2:.4f}".format( np.nanmean(chisqr_all[:,T]), chisqr_all.shape[0], np.nanmedian(chisqr_all[:,T])/chi2.ppf(0.5, df=1) ))
    return

#%% setup and flags
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test for association with FMA (step 2). \nCan be invoked as \n \t`python fma_step2.py -g geno -C C -r residuals --calib calibrators --covar covariates --minMAF 1e-5 --out prefix`", formatter_class=RawTextHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bgen", help="prefix for bgen/sample files", type=str)
    group.add_argument("--bfile", "-g", help="prefix for bed/bim/fam files", type=str)
    parser.add_argument("--chr", "-C", help="chromosome to test for", type=int)
    parser.add_argument("--residuals", "-r", help="file with LOCO residuals for current chromosome (saved in step 1)", type=str)
    parser.add_argument("--calib", help="file with calibrators (saved in step 1)", type=str)
    parser.add_argument("--covar", help="file with covariates; should be in \"FID,IID,Var1,Var2,...\" format and tsv", type=str, default=None)
    parser.add_argument("--minMAF", help="lower threshold for allele frequency", type=float, default=1e-4)
    parser.add_argument("--batchSize", help="number of variants for each batch", default=1000, type=int)
    parser.add_argument("--output", "-o", help="prefix for where to save any results or files", type=str)
    # parser.add_argument("--PoolSize", help="number of batches to process in parallel", default=5, type=int)
    parser.add_argument("--nThreads", help="number of computational threads to be used", default='1', type=str)
    parser.add_argument("--debug", help="print intermediate info", action="store_true")
    
    # WARNING: I think these should be called prior to importing numpy!
    args = parser.parse_args()
    os.environ['MKL_NUM_THREADS'] = args.nThreads
    os.environ['OPENBLAS_NUM_THREADS'] = args.nThreads
    os.environ["OPENBLAS_NUM_THREADS"] = args.nThreads
    
    time0 = time.perf_counter()
    header = True # we need this for printing the header only once
    if args.debug:
        debug=True
        print("FMA - run with options:")
        print(', '.join(f'{k}={v}' for k, v in vars(args).items()))
    else:
        debug=False
    
    print("\nFMA: Calculate SNP test statistics using pre-computed residuals\n" )
    
    if args.bfile is not None:
        if os.path.exists( args.bfile ):
            print("Genotypes will be loaded from", args.bfile)
            snp_on_disk = Bed( args.bfile, count_A1=True )
            use_bed = True
    elif os.path.exists( args.bgen ):
        if os.path.exists( args.bgen ):
            print("Genotypes will be loaded from", args.bgen)
            snp_on_disk = Bgen(args.bgen).as_snp(max_weight=2)
            use_bed = False
    else:
        print("ERROR: missing genotype files.")
        use_bed = False
    
    # create a hashmap from true-ID to {0,...,N-1}
    samples_index = np.zeros((snp_on_disk.iid_count,2), dtype=int)
    for i,x in enumerate(snp_on_disk.iid):
        samples_index[i,:] = int(x[1][:7]), i
    samples_index = pd.DataFrame( samples_index )
    samples_index.set_index( samples_index[0], inplace=True )
    
    df = pd.read_csv( args.calib, sep='\t', header=None )
    calibrators = {}
    for i in range(len(df)):
        calibrators[ str(df.iloc[i,0]) ] = 1
    N_phen = len(calibrators)
    pheno_names = list(calibrators.keys())
    
    if args.covar == None:
        # I'll assume no covariates are needed; only an intercept will be used
        print("No covariates are passed - will only use an intercept")
        covars = pd.DataFrame(samples_index)
        covars["intercept"] = 1
    else:
        covars = pd.read_csv( args.covar, sep='\s+', index_col="FID" )
        covars["intercept"] = 1
    
    # initialise files with sum-stats
    for T in range(N_phen):
        if os.path.exists(".".join( [args.output, pheno_names[T],"sumstats.gz"])):
            os.remove( ".".join( [args.output,pheno_names[T],"sumstats.gz"]) )
                
    if use_bed:
        #TODO: add covariates!
        chr_map = find_chromosomes( args.bfile )
        
        for C,CHR in enumerate( chr_map ):
            # read the residuals for this chrom
            print("Chrom -",C+1, end=". ")
            Traits = pd.read_csv( "{0}.loco{1}.residuals.gz".format(args.residuals, C+1), sep='\s+', index_col="FID" )
            
            samples = np.intersect1d( samples_index.index, Traits.index ) 
            samples = np.intersect1d( samples, covars.index )
            sample_index_to_use = samples_index.loc[samples][1].values # TODO; is there a bug here?
            
            W = covars.loc[samples].iloc[:,1:].to_numpy()
            Traits = Traits.loc[samples]
            residuals = Traits.iloc[:,1:].to_numpy()
            
            FMA_test_chrom( C, 
                   snp_on_disk[:, chr_map[CHR]], 
                   residuals, calibrators, W, 
                   sample_index_to_use, 
                   min_maf=args.minMAF, 
                   batch_size=args.batchSize, 
                   saveas=args.output, #+".chr"+str(args.chr), 
                   verbose=False )
    else:
        
        Traits = pd.read_csv( args.residuals, sep='\s+', index_col="FID" )
        
        samples = np.intersect1d( samples_index.index, Traits.index ) 
        samples = np.intersect1d( samples, covars.index )
        sample_index_to_use = samples_index.loc[samples][1].values # TODO; is there a bug here?
        print( "Effective sample size:", len(sample_index_to_use) )
        
        Traits = Traits.loc[samples]
        residuals = Traits.iloc[:,1:].to_numpy()
        W = covars.loc[samples].iloc[:,1:].to_numpy()
        assert residuals.shape[1] == N_phen, "Shape of residuals is not as expected!"
        del Traits, covars
                
        FMA_test_chrom( args.chr, 
                   snp_on_disk, 
                   residuals, calibrators, W, 
                   sample_index_to_use, 
                   min_maf=args.minMAF, 
                   batch_size=args.batchSize, 
                   saveas=args.output, #+".chr"+str(args.chr), 
                   verbose=False )
    
    print("FMA total time for test-statistics = {0:.2f} secs".format( (time.perf_counter()-time0) ))
# end-of-file