#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for the streaming version of FMA, for multiple trait support.
Depends on FMA_main: here are only those functons specific to the streaming approach 
and any alterations of the rest. 

Created on Fri Apr 16 12:39:04 2021
@author: kalantzi
"""

import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed
from scipy.sparse import diags
from multiprocessing.pool import ThreadPool as Pool
import time, h5py, os.path
from . import find_chromosomes, var_chunked, dot_chunked, printLog #, numba_dot_chunked

save_temp_res = False # feature for checking at intermediate estimates of the pheno residuals

def FMA_streaming_mt( plinkfile, Traits, samples, annotation, VC, arg_uniform=False, arg_Ncal=3, cores=4, cg_max_iters=50, arg_debug=False, useHDF=None ):
    """
    Randomised Complex Trait Association - for multiple genetic components and many traits. 
    Step 1: Calculate the LOCO residuals and the calibration factor.
    Should be followed by `FMA_JustTest` to get the test statistics. We neither get those for LR here anymore. 
    Variants will be indexed according to genomic position.
    INPUT:  0) Traits: N x Nt dataframe with phenotypes; should have a header
            1) VC: (Nc+1) x Nt array with VC for each phenotype
            2) the rest are as before
    OUTPUT: 0) dataframe with ['BETA','CHISQ','P'] for Linear Regression
            1) calibration as estimated from `N_calibr`non-causal SNPs
            2) N_chr sets of residuals
            3) other stuff, e.g. prospective statistics for the N_calibr SNPs
    """
    print("="*56)
    # the next are variables to be used by each spawn process during the core calculations
    global N_phen, N_chr, N_calibr, nrhs, nrhs_chrom, nrhs_chrom_pheno, N, M_all, batch_size
    global samples_to_use, bSizeA, bSizeB, uniform, hdf5, chr_map, block_info_all, debug, readings
    debug = arg_debug
    residuals_all = {}; calibrators = {}; # the final output

    printLog("FMA: Multiple Components + Many traits + Fast Streaming with Multi-Processing \n\n1. Initializations")
    if cores == '1':
        print("Will only process 1 block per genotype. Consider using more with `--nCores` for a speed-up.")
    else:
        print("Will be processing",cores,"batches in parallel.")
    time0 = time.perf_counter()
    snp_on_disk = Bed(plinkfile, count_A1=True)
    N, M_all = snp_on_disk.shape 
    chr_map = find_chromosomes(plinkfile); N_chr = len(chr_map)
    if samples == None:
        samples = range(N)
    else:
        N = len(samples)
    
    ### parameters that affect efficiency ###    
    bSizeA = 2**25 // N
    bSizeB = 5000
    # cores: controls the number of geno_blocks that are processed simultaneously, should be <= n_CPU_cores
    batch_size = cores # controls the number of blocks to be processed from the same pool of workers
    # the batch_size should ideally be equal to the total number of blocks. Note that we need to keep `batch_size` arrays 
    # of size `N x nrhs` floats in memory, so the higher this is the more RAM we use. On the other hand, the smaller the batch size, 
    # the more pools of processes we'll need to create, so batch_size should be larger than the pool size 
    # (the cost of initialisation is probably trivial wrt the processing time of each block)
    
    samples_to_use = samples # this follows the BED index
    if Traits.shape[0] != len(samples): 
        print("\nWARNING: number of phenotyped samples doesn't match the fam size!")
        Traits = Traits.loc[ Traits.index[samples] ] 
    
    # remove traits with unreasonably low Ve estimates
#     good_traits = np.where( VC[-1,:]>=0.20 )[0] 
    bad_traits = list(Traits.columns[ np.where( VC[-1]<0.15 )[0] ])
    if len(bad_traits) > 0:
        print("\nWARNING:", *Traits.columns[np.where( VC[-1,:]<0.15 )[0]], "might fail to converge and will be excluded from the LMM analysis!")
        
    # remove traits with h2=0 from the analysis, but keep them for LR
    bad_traits.extend( Traits.columns[ np.where( VC[-1]==1 )[0] ] )
    if len(bad_traits) > 0:
        print("WARNING: The following traits have h2=0 and will be excluded from the LMM analysis!", *bad_traits)        
        VC = VC[:, np.setdiff1d( np.arange(Traits.shape[1]), np.where( VC[-1]==1 )[0] ) ] # find which columns to keep for VC
        Traits_bad = pd.DataFrame()
        for x in bad_traits:
            Traits_bad[x] = Traits[x]
            Traits.drop(x, axis=1, inplace=True)

    print("\nGetting SNP mean/std info:")
    col_means, col_vars = streaming_preparation_mt( snp_on_disk, int(M_all/500), Traits, samples, debug=debug )
    
    # MAF control comes here; we consider SNPs with MAF >=1e-4
    bad_snps = np.where(col_means<=2e-4)[0] # TODO: add a flag for a user-defined threshold?
    if len(bad_snps)>0: 
        print("\nWARNING: There are {0} monomorphic/ultra-rare markers which will be excluded.".format(len(bad_snps)))
        print(bad_snps[:5])
        
    N_phen = Traits.shape[1] # as we dont pass FID+IID
    N_comp = VC.shape[0]-1
    printLog("\n{0} samples, {2} chromosomes, {3} components, and {4} traits.".format(N, M_all, N_chr, N_comp, N_phen))

#    block_ID_all = find_components( annotation, N_comp, chr_map )
    block_ID_all = find_components_chunked( annotation, chr_map, bad_snps, N_comp )
    if sum([ len(x) for x in block_ID_all.values()]) != M_all: # | ( int(sum(component_sizes))!=M_all*N_comp):
        print("\nWARNING: the number of annotated variants is different than the total number of SNPs:", end=' ')
        print( sum([ len(x) for x in block_ID_all.values()]) ) #, sum(component_sizes) )
        M_all = sum([ len(x) for x in block_ID_all.values()])

    if debug:
        print("Variance component estimates:")
        for T in range(N_phen):
            print(*VC[:,T]) # print all the values without brackets
    
    if arg_uniform: # determine what weights will be used in `Covar_matvec_MC`
        print("Equal weights will be used in the formation of each LOCO GRM.")
    else:
        print("The weights in the formation of each LOCO GRM will be as `M_k - M_k,C`.")
    uniform = arg_uniform
    N_calibr = arg_Ncal
    print(N_calibr,"SNPs will be used per chromosome to estimate the calibration factor.")
        
    if useHDF == None:
        # will not create a new HDF5 file
        hdf5 = annotation+".hdf5"
    else:
        # will use the given filename for the HDF5
        hdf5 = useHDF
        
    if not os.path.isfile( hdf5 ): # check if one exists already 
        if debug: print("HDF5 was not found.")
        convert_bed_to_hdf5(plinkfile, block_ID_all, samples, hdf5, chunk_size=1000)
    else:
        print("HDF5 file to use for streaming:", hdf5)


    # select all the non_causal variants that will be used for calibration
    RHS = np.zeros((N, N_phen*(N_chr + N_calibr*N_chr))) # for all the systems we will solve Nt*Nc*(1+Ncalibr)
    # example for N_calibr=2:
    # [ [y1, x1;1, x1;2], [y1, x2;1, x2;2], ..., [y1, x22;1, x22;2]] and repeat for y2, y3, ... yT
    print("Getting non-causal variants for estimating the calibration factors...", end=' ')
    col=0
    for T in range(N_phen):
        for C in range(N_chr): # NOT equivalent to `for C in chr_map.keys():` as we follow index {0..21}
            RHS[:, col] = Traits.iloc[:,T]; col +=1
            if N_calibr > 0:
                temp, _ = find_noncausal( snp_on_disk, Traits.iloc[:,T], samples, chr_map[C], N_calibr )
                for i in range(N_calibr):
                    RHS[:, col] = temp[:,i]
                    col += 1
#            indx = np.random.permutation( np.where( df_LR[ chr_map[C], T ] < 5 )[0] )[:N_calibr] # indx runs in {0,1,...,M_CHR} range
#             for v in range(N_calibr):
#                 RHS[:, col] = (snp_on_disk[ samples, chr_map[C][indx[v]] ].read().val - col_means[chr_map[C][indx[v]]]).reshape(-1)
#                 col += 1
        
    nrhs  = RHS.shape[1]
    nrhs_chrom = N_phen*(N_chr-1)*(1+N_calibr) # the total Nupdates per chromosome
    nrhs_chrom_pheno = (N_chr-1)*(1+N_calibr) # the Ncolumns to update for one trait+one chrom
    print(" {0} RHS in total are set for the linear systems.".format(nrhs))
     
    # we'll use a dict to map each (k,C) ID to the corresponding "block_info" structure 
    block_info_all = {}
    right=0
    for block in block_ID_all: # block="(comp, chrom)"
        block_info_all[block] = block_info(block, block_ID_all[block], right, VC[block[0],:] )
        right += len( block_ID_all[block] )
        # TODO: is it likely to have a sigma << 0 ?
        block_info_all[block].means = col_means[ block_ID_all[block] ]
        block_info_all[block].weights = col_vars[ block_ID_all[block] ]**(-1)
    del col_means, col_vars, block_ID_all # not needed anymore
    print("\nInitializations are complete. Duration for this step = {0:.2f} mins.".format( (time.perf_counter()-time0)/60 ) )

    ##################################
    ## Core computations start here ##
    
    print("\n2. Core calculations")
    time1 = time.perf_counter()
    readings = 0
    # Conjugate Gradients
    threshold = np.linalg.norm(RHS, ord=2, axis=0)*5e-5
    CG_pack, norms, q = solve_conj_grad( RHS, VC, threshold, cg_max_iters )

#    CG_pack, norms, q = solve_momentum( RHS, VC, tol=5e-4, max_iters=30)
#    CG_pack, norms, q = solve_adam( RHS, VC, tol=5e-4, max_iters=30)
    
    ## Core calculations end here ##        
    ################################
    print("CG is done; Total time for {1} systems and {2} iterations: {0:.2f} mins.".format( (time.perf_counter()-time1)/60, CG_pack.shape[1], q+1) )
    print("Total number of readings:",readings)
#     if len(not_converged)==0:
#         print("Every case has converged :)")
#     else:
#         print("Cases that did not converge: ", *not_converged.values())
#     assert len(converged)>0, "No cases have converged!"
    
    print("\n3. Final calculations for the calibration factors" )
    
    if N_calibr > 0:
        # Get the estimates for GRAMMAR-gamma; work in a trait-based structure and keep the structure
        col=0; 
        for T in range(N_phen):
            # we should ideally go over a dictionary of converged cases which has the form "trait_index:trait_name"
            # TODO: be careful and skip non-converged cases accordingly
            print("\nWorking for {0} with total h2={1:.3f}".format(Traits.columns[T], 1-VC[-1,T]))
    
            prospectiveStat = []; uncalibratedRetrospectiveStat = [] ## as in bolt  
            gamma = []
            # unpack the results from CG; these are ordered as [chr1-res, chr1-x1, ..., chr2-res, chr2-x1,..., chr22-x3]
            col = T*(N_chr + N_calibr*N_chr)
            for C in range(N_chr):
                Vinv_y = CG_pack[:,col]; 
                col += 1
                for i in range(N_calibr):
                    Xm = RHS[:, col] # get the original SNP
                    temp = (Xm.T.dot(Vinv_y))**2
                    # prospective = N * (x^T V^-1 y)^2 / (x^T V^-1 x * y^T V^-1 y)
                    prospectiveStat.append( N*temp/ Xm.T.dot(CG_pack[:,col]) /np.dot(Traits.iloc[:,T], Vinv_y) )
                    # TODO: replace N with N-C, adjusting for the number of covariates
                    # prospectiveStat.append( temp/Xm.T.dot(CG_pack[:,col]) ) # this is probably wrong but similar to the original formula
                    uncalibratedRetrospectiveStat.append( N*temp/Xm.T.dot(Xm)/Vinv_y.dot(Vinv_y) ) # norm(Xm)^2 * norm(VinvY)^2
                    gamma.append( Xm.T.dot(CG_pack[:,col])/Xm.dot(Xm) )
                    col += 1
            print("Grammar-gamma approximation = {0:.4f} ({1:.4f})".format( np.mean(gamma), np.std(gamma)) )
        
            calibration = (sum(uncalibratedRetrospectiveStat) / sum(prospectiveStat)).item()    
            temp = np.array(uncalibratedRetrospectiveStat)/np.array(prospectiveStat)
            print("AvgPro: {0:.3f} AvgRetro: {1:.3f} Calibration: {2:.3f} ({3:.3f}) ({4} SNPs).".format(np.mean(prospectiveStat), np.mean(uncalibratedRetrospectiveStat), calibration, np.std(temp), len(prospectiveStat)))
            print("Ratio of medians: {0:.4f} | Median of ratios: {1:.4f}".format(  np.median(uncalibratedRetrospectiveStat)/np.median(prospectiveStat), np.median(temp))   )
            print("Mean, median and std of ratios = {0:.6f}, {1:.6f}, {2:.6f}".format(np.mean(temp), np.median(temp), np.std(temp) ))
            
            calibrators  [ Traits.columns[T] ] = calibration
            if norms[T*N_chr*(1+N_calibr)]>threshold[T*N_chr*(1+N_calibr)]:
                print("WARNING: CG has not converged for " + Traits.columns[T])
    else:
        print("WARNING: Using 1 as calibration constants!")
        for T in range(N_phen):
            calibrators[ Traits.columns[T] ] = 1
        
    # Finally, create a chrom-based dict for the residuals
    for C in range(N_chr):
        residuals_all[C] = np.zeros( (Traits.shape[0], Traits.shape[1] + len(bad_traits)) )
        for T in range(N_phen):
            col = T * N_chr * (1+N_calibr) + C * (1+N_calibr)
            residuals_all[C][:,T] = CG_pack[:,col]
        for x in bad_traits:
            T += 1
            residuals_all[C][:,T] = Traits_bad[x]

    for x in bad_traits:
        calibrators[x] = 1 # add a dummy calibr factor for any traits with h2=0
        
    print("\nModel fitting is done! Total duration for {0} traits = {1:.2f} mins (plus trait preparation).".format( N_phen, (time.perf_counter()-time0)/60 ))
    print("="*45)
    return [residuals_all, calibrators, q]
  
#%% Support functions; only specific to streaming    
class block_info:
    def __init__(self, label, index, starting_from, vc):
        self.label = label
        # self.index = index
        self.index = np.arange(starting_from,starting_from+len(index) )
        self.size  = len(index) # this is for the math within XX'Z, not for the final weight
        self.sigma = vc
        self.compo = label[0]
        self.chrom = label[1]
        self.chunk = label[2]
        self.means = np.empty( self.size )
        self.weights = np.empty( self.size )
# TODO: add a function here that loads the corresponding block of genotypes?
   
def convert_bed_to_hdf5(plinkfile, block_ID, samples, save_as, chunk_size=1000):
    """Based on Hrushi's code but without phenotypic data and SNP-based partitioning.
    The hdf5 is structured as [(K1,C1)]
    """
    snp_on_disk = Bed(plinkfile, count_A1=True)
    _,M = snp_on_disk.shape
    N = len(samples)
    h = h5py.File( save_as, 'w')
    dset1 = h.create_dataset('genotypes', (M, N), chunks=(500, N), dtype= 'int8')
    # if K==1:
    #     print("Creating the hdf5 file as the bed file:")
    #     for i in range(0, M, chunk_size):
    #         dset1[i:min(i+chunk_size, M), :] = snp_on_disk[ :, i:min(i+chunk_size, M)].read( dtype='int8' ,  _require_float32_64=False ).val.T
    # else:
    print("Creating the hdf5 file on a component-based order, named as", save_as)
    left=0; right=0
    for b in block_ID:
        right += len(block_ID[b])
        X = snp_on_disk[ samples, block_ID[b] ].read( ).val #dtype='int8' ,  _require_float32_64=False
        # replace any nans with the median:
        X = np.where( np.isnan(X), np.nanpercentile(X, 50, axis=0, interpolation="nearest"), X )
        dset1[left:right, :] = np.array( X.T, dtype='int8' )
        left += len(block_ID[b])
    h.close()
    return

def find_noncausal( snp_on_disk, trait, samples, variants, N_calibr=2, t_min=1, t_max=5 ):
    # variants should be in the {0,1,...,M-1} index
    #  samples should be in the {0,1,...,N-1} index
    varY = trait.dot(trait)
    X = np.zeros( (len(samples),N_calibr) )
    indx = np.random.permutation(variants)
    scores = np.zeros(N_calibr)
    s = 0
    for v in indx:
        g = np.nan_to_num( snp_on_disk[ samples, v ].read().val.reshape(-1), nan=0 )
        # TODO: replace this with proper meadin-imputed nans
        g = g - np.mean(g)
        varG = g.dot(g.T)
        if varG>0: # for the case we've sub-sampled and there's no variation
            test = (len(samples)-2)/( varY*varG/(g.T.dot(trait))**2 - 1 )
            if t_min < test <= t_max:
                X[:,s] = g
                scores[s] = test
                s += 1
        if s>=N_calibr:
            break
    
    return X, scores

def streaming_preparation_mt( data_on_disk, N_batches, Traits, samples, W=None, debug=False ):
    N, M = data_on_disk.shape
    col_means = np.empty(M); col_vars = np.empty(M)
    BS = M//N_batches # number of columns per batch
#     df_LR = np.zeros( (M, Traits.shape[1])) # this will be of size M x Nt
    if debug:   
        # just give random values to proceed
        col_means = 0.4*np.ones(M); 
        col_vars = 0.1*np.ones(M)
    else:
        print("This will take a few minutes as we will process",N_batches,"batches. Processed so far:")
        for i in range(0, N_batches+1):
            if i%10==0: print("B"+str(i), end=', ')
            index = range(i*BS, min((i+1)*BS,M))
            X = data_on_disk[ samples,index ].read().val # dtype='int8' ,  _require_float32_64=False
            # replace any nans with the median:
            X = np.where( np.isnan(X), np.nanpercentile(X, 50, axis=0, interpolation="nearest"), X )
            col_means[index] = np.nanmean(X, axis=0)
            col_vars[index] = var_chunked(X, blocks=4) 
            # since the next part was very slow, I'm now skipping the LinReg part and only get N_chr * N_calibr variants later 
#             if not debug:
#                 X0 = X - col_means[index] 
#                 for T in range(Traits.shape[1]):
#                     # test in batches and keep only the CHISQ statitics:
#                     df_LR[index,T] = MyLinRegr( X0, Traits.iloc[:,T], W)["CHISQ"]
    return np.array(col_means), np.array(col_vars) # df_LR

def geno_blocks( snp_indices, sample_indices, hdf5 ):
    """
    An intermediate step for loading a specific block of genotypes. 
    """
    h5_file = h5py.File(hdf5 , 'r')
    temp = h5_file['genotypes'][ snp_indices, : ]
    if h5_file['genotypes'].shape[1] == len(sample_indices):
        return temp[:, :] # issue with fancy indexing otherwise
    else:
        return temp[:, sample_indices]
#    return snp_on_disk[ sample_indices, snp_indices ].read( dtype='int8' ,  _require_float32_64=False ).val

def find_components_chunked( annotation, chr_map, bad_snps, N_comp, BS=3000 ):
    """
    This function creates an index of chrom-wise SNP indices for each component, assuming that a class 
    can span multiple chromosomes, and then each chrom is partitioned to chunks for efficiency.
    """
    df = pd.read_csv(annotation, sep=' ', header=None)
    assert df.shape[1] == N_comp, "Annotation file and VC size mismatch!"
    df.iloc[bad_snps,:] = 0
    class_map = {}; 
    print("Component sizes: ", end='')
    i_all = 0 # across chrom batch index
    total = 0; 
    for k in range(N_comp):
        comp_id = np.where( df[k]==1 )[0]
        i_comp = 0
        total_comp = 0
        for C in chr_map:
            comp_chr = np.sort( np.intersect1d( comp_id, chr_map[C] ))
            start = 0
            i_c = 0 # within chrom batch index
            while start < len(comp_chr):
                index = comp_chr[start:start+BS]
                class_map[(k,C,i_c)] = np.intersect1d( index, comp_chr )
                total += len(class_map[(k,C,i_c)])
                start += len(index) # index[-1]
                i_c += 1
                i_all += 1
            i_comp += i_c
            total_comp += sum([ len(class_map[(k,C,i)]) for i in range(i_c) ])
        print("{0}:({1},{2}), ".format(k,i_comp,total_comp),end='')
    print("\nOverall: {0} variants, {1} chunks.".format(total,i_all)) # return carrier
#     assert total == sum([len(x) for x in chr_map.values()])
    return class_map
  
#%% Low level/Math functions:
       
def solve_conj_grad( RHS, VC, threshold, max_iters=50 ):
    """
    Solves systems of the form V*X = RHS using the conjugate gradients method, for V formed according to VC.
    Bottleneck: the number of iters is proportional to N, otherwise this is optimal for our purposes.
    """
    global printLog, bSizeA, bSizeB, Z, samples_to_use
    # Initialization
    solutions = np.zeros(RHS.shape, dtype=np.float64) # this will converge to the solution
    residuals = RHS.copy() # initialization, R(0) = B-A*X
    Z = RHS.copy() # the "search directions", this is what we'll multiply the covar matrix with
    nrhs  = RHS.shape[1]
    r2old = np.zeros((nrhs,1), RHS.dtype)
    alpha = np.zeros((1,nrhs), RHS.dtype) # for elementwise
    beta  = np.zeros((1,nrhs), RHS.dtype) # for elementwise

    for q in range(max_iters): # or until convergence
        
        tic = time.perf_counter()
        
        VZ = covar_matmat(Z, VC)
        
        # check the residuals and update w.r.t CG
        for j in range(nrhs):
            r2old[j] = residuals[:,j].dot( residuals[:,j] )
            alpha[0,j] = r2old[j] / Z[:,j].dot( VZ[:,j] ) 
        solutions += alpha * Z # update the solutions; elementwise
        residuals -= alpha * VZ 
        norms = np.linalg.norm(residuals, ord=2, axis=0)
        if (norms <= threshold).all(): # TODO: avoid the extra calculations in some cases (more than necessary)
            break # is this the proper condition? we end up with more iterations otherwise
        for j in range(nrhs):
            beta[0,j] = residuals[:,j].dot(residuals[:,j].T) / r2old[j]
        Z = residuals + beta * Z # update Z; elementwise
        # BOLT prints: iter 11:  time=18.29  rNorms/orig: (7e-07,0.0005)  res2s: 13277.4..13999.5
        printLog("CG iteration {0}: time={1:.2f}, mean norm: {2:.6f}. Block sizes: {3} , {4}.".format(q+1, time.perf_counter()-tic, np.nanmean(norms),  bSizeA, bSizeB ), flush=True)
    
        ######################################################
        # save intermediate residuals; useful for developement
        if save_temp_res:
            if q>=20 and q%5==0:
                for T in range(N_phen)[:5]:
                    df = pd.DataFrame()
                    df['FID'] = samples_to_use
                    df['IID'] = samples_to_use
                    col = T*(N_chr + N_calibr*N_chr)
                    for C in range(N_chr):
                        df["LOCO"+str(C+1)] = solutions[:,col]; 
                        col += 1+N_calibr
                    df.to_csv( "fma.{2}k.temp_residuals.{0}.pheno{1}.gz".format(q, T, N//1000), index=None, sep='\t', compression='gzip')
        ######################################################
    return solutions, norms, q

def solve_adam( RHS, VC, tol=5e-3, max_iters=50):
    """
    Solves systems of the form V*X = RHS using the ADAM solver, for V formed according to VC.
    Sub-optimal as the hyper-parameters need tuning, and the implementation is likely bad.
    """
    alpha = 0.3
    beta1 = 0.90
    beta2 = 0.99
    eps = 1e-8
    global Z
    
    Z = np.zeros(RHS.shape, dtype=np.float64) 
    # Z is what will hold the solutions for the linear systems (and will be mulitplied with V, the covar matrix)
    D_old = np.zeros(RHS.shape, dtype=np.float64)
    H_old = np.zeros(RHS.shape, dtype=np.float64)
    print( "ADAM: alpha = {0}, beta1 = {1}, beta2 = {2}.".format( alpha, beta1, beta2 ))
    for q in range(max_iters): # or until convergence
        tic = time.perf_counter()
        
        grad = covar_matmat( Z, VC ) - RHS 
        
        D_new = beta1*D_old + (1-beta1) * grad
        H_new = beta2*H_old + (1-beta2) * grad**2
        Z -= alpha * D_new/(1.0 - beta1**(q+1)) / (np.sqrt(H_new/(1.0 - beta2**(q+1))) + eps)
        D_old = D_new
        H_old = H_new
        norms = np.linalg.norm(grad, ord=2, axis=0)
        printLog("Iteration {0}: time={1:.2f}, mean norm: {2:.6f}. Block sizes: {3} , {4}.".format(q+1, time.perf_counter()-tic, np.nanmean(norms),  bSizeA, bSizeB ), flush=True)
        if (norms <= tol).all(): # TODO: avoid the extra calculations in some cases (more than necessary)
            break
          
    return Z, norms, q
  
def solve_momentum( RHS, VC, tol=5e-3, max_iters=50):
    """
    Solves systems of the form V*X = RHS using the steepest descend, for V formed according to VC.
    Sub-optimal as the hyper-parameters need tuning, and the implementation is likely bad.
    """
    alpha = 0.7
    beta = 0.3
    global Z
    
    Z = np.zeros(RHS.shape, dtype=np.float64) 
    # Z is what will hold the solutions for the linear systems (and will be mulitplied with V, the covar matrix)
    Dold = np.zeros(RHS.shape, dtype=np.float64)
    print( "alpha = {0}, beta = {1}".format( alpha, beta ))
    for q in range(max_iters): # or until convergence
        tic = time.perf_counter()
        
        grad = covar_matmat( Z, VC ) - RHS 
        
        Dnew = beta*Dold - (1-beta) * grad
#        Z += alpha*(q+1)**(-1/3) * Dnew # using a simple decay; needs improvement TODO
        Z += alpha * Dnew # fixed learning rate
        Dold = Dnew #.copy()
        norms = np.linalg.norm(grad, ord=2, axis=0)
        printLog("Iteration {0}: time={1:.2f}, mean norm: {2:.6f}. Block sizes: {3} , {4}.".format(q+1, time.perf_counter()-tic, np.nanmean(norms),  bSizeA, bSizeB ), flush=True)
        if (norms <= tol).all(): # TODO: avoid the extra calculations in some cases (more than necessary)
            break
          
    return Z, norms, q
  
def covar_matmat(X, VC):
    """
    Calculates the V*X, where V is formed according to VC, using multiprocessing.
    """
    global N_phen, N_chr, N_calibr, readings, block_info_all, batch_size

    VX = np.zeros(X.shape, dtype=np.float64)
    for T in range(N_phen):
        upd_indx_tmp = np.arange( T*N_chr*(N_calibr+1), (T+1)*N_chr*(N_calibr+1))
        VX[:,upd_indx_tmp] = VC[-1,T]*X[:,upd_indx_tmp] # the environmental component, shared across all cases
        
    for i in range(len(block_info_all)//batch_size+1):
        index = range(i*batch_size, min((i+1)*batch_size,len(block_info_all)))
        POOL = Pool(batch_size)
        # POOL = multiprocessing.Pool(cores)
        partial_VX = POOL.map( core_calc_mp, (b for b in [ list(block_info_all.values())[x] for x in index]))
        # ideally each worker should write directly on `VX` but I can't find a way to do that
        # TODO: thus I always keep `batch_size` matrices in the `partial_VX` list
        POOL.close (); POOL.join () # close; do we need this??      
        readings += len(partial_VX)
        for x in partial_VX:    
            VX += x
        
    return VX
  

def core_calc_mp(block):
    """
    Calculate XX'Z, and adjust with the corresponding weight, where X is the standardized form of block G and Z is defined by the solver.
    This is the hardest part of the framework and its highly optimised to avoid unnecessary casting or intermediate arrays.
    Some hyperparameters are `bSizeA`, `bSizeB`, which control the sizes for block-wise calculations, 
    and `uniform` which controls what weights we'll multiply the GRMs with.
    """
    global N_phen, N_chr, N_calibr, nrhs, nrhs_chrom, nrhs_chrom_pheno, N, M_all
    global samples_to_use, bSizeA, bSizeB, uniform
    if debug: printLog("hi", end=' ')

    # find which columns/sums this block needs to be included (all besides those of block.chrom):
    update_index = [True]*nrhs # # this includes all the traits
    for T in range(N_phen):
        for i in np.arange( block.chrom*(N_calibr+1), block.chrom*(N_calibr+1)+N_calibr + 1):
            update_index[ i + T*N_chr*(1+N_calibr) ] = False
#     BS = 500* 2**17 // G.shape[0] # so as each chunk takes less than 500MB of RAM for casting to float -- could be similar to bSizeB
#    N_batches = N//bSizeB if N%bSizeB==0 else N//bSizeB+1  # params for N-based calculations
    
    Z_sum = np.sum(Z[:,update_index], axis=0).reshape(1,nrhs_chrom)
    
    G = geno_blocks( block.index, samples_to_use, hdf5 ) # read from disk 
    # note: `geno_blocks` returns `SNPs x samples` format

    tic0 = time.perf_counter()
    if debug: time.sleep(1)
    tic1 = time.perf_counter()
    XZ = dot_chunked( G, Z[:,update_index], bSizeA ) # mat-mat ONE 
#    XZ = numba_dot_chunked(G, Z[:,update_index], 2000)  # mat-mat ONE
    if debug: printLog("MatMat1: {0:.2f}.".format(time.perf_counter()-tic1), end=' ')
    temp1 = diags(block.weights).dot(XZ) # D*X*Z
    del XZ # not needed anymore (new)
    
    temp2 = diags(block.weights).dot( block.means.reshape(block.size, 1).dot(Z_sum) )  # D \mu 1 U
    
    tic1 = time.perf_counter()
    XDXZ = dot_chunked( G.T, temp1 - temp2, bSizeB) # mat-mat TWO
    #    XDXZ += numba_dot_chunked(G.T, (temp1 - temp2), bSizeB)  ## TERM-1 + TERM-3
    if debug: printLog("MatMat2: {0:.2f}.".format(time.perf_counter()-tic1), end=' ')
    
    XDXZ -= np.ones((G.shape[1], 1)) * block.means.T.dot(temp1)  ## TERM-2
    XDXZ += block.means.T.dot( diags(block.weights).dot(block.means.reshape(block.size, 1).dot(Z_sum)) )  ## TERM-4
    
#    # if debug: time.sleep(1)
#    # the next produces a small spike in RAM
#    # XDXZ -= np.ones((N,1))*block.means.T.dot( temp ) # term-2; we use temp instead of diags(block.weights).dot(XZ)
#    for i in range(N_batches):
#        index = range(i*bSizeB, min((i+1)*bSizeB,N))
#        XDXZ[index,:] -= np.ones((len(index),1))*block.means.T.dot( temp )
#    
#    # prepare and add subtract the third term
#    temp = diags( block.weights ).dot( block.means.reshape(block.size,1).dot(Z_sum)) # D*mu*1*Z
#    tic1 = time.perf_counter()
#    # mat-mat THREE comes here:
#    # dot_chunked_sub( G.T, temp, XDXZ, G.shape[1]//1000)  
#    for i in range(0, N_batches+1):
#        index = range(i*bSizeB, min((i+1)*bSizeB,N))
#        XDXZ[index, :] -= G.T[index,:].dot(temp)
#    if debug: printLog("MatMat3: {0:.2f}.".format(time.perf_counter()-tic1), end=' ')
#        
#    # I thought that the direct assignment produces a spike in memory, so I tried a 2-step update
#    XDXZ += np.ones((N,nrhs_chrom))*block.means.T.dot( diags(block.weights).dot(block.means.reshape(block.size,1).dot(Z_sum))) # part of term-4
#    temp1 = block.means.T.dot( diags(block.weights).dot(block.means.reshape(block.size,1).dot(Z_sum)))
#    for i in range(N_batches):
#        index = range(i*bSizeB, min((i+1)*bSizeB,N))
#        XDXZ[index,:] += temp1

    # now form the LOCO sums ("V*Z") and complete the CG iteration
    tic1 = time.perf_counter()
    # each block.sigma is now a vector of values and each set of columns needs to be assigned to a different one
    # first we need to get M_k by looking at all the blocks of current component
    weights = np.repeat( sum( [ b.size for b in block_info_all.values() if b.compo==block.compo ] ), nrhs_chrom_pheno)
    # then for each chrom, update the corresponding weights accordingly
    chroms = np.setxor1d( range(N_chr), block.chrom )
    for i,C in enumerate(chroms):
        M_kc = sum([ b.size for b in block_info_all.values() if b.compo==block.compo and b.chrom==C ])
        weights[np.arange( i*(N_calibr+1), i*(N_calibr+1)+N_calibr + 1)] -= M_kc # assign the same number multiple times
    
    VZ = np.zeros( (N,nrhs), dtype=np.float64)
    for T in range(N_phen):
        # all but block.chrom for trait T;
        indx_trait = np.arange( T*N_chr*(1+N_calibr), (T+1)*N_chr*(1+N_calibr))
        indx_chrom = np.arange( T*N_chr*(1+N_calibr) + block.chrom*(N_calibr+1), T*N_chr*(1+N_calibr) + (block.chrom+1)*(N_calibr+1))
        upd_indx_1 = np.setxor1d(indx_trait, indx_chrom)
        upd_indx_2 = np.arange( T*nrhs_chrom_pheno, (T+1)*nrhs_chrom_pheno )
        if uniform: 
            VZ[:,upd_indx_1] += XDXZ[:,upd_indx_2]*block.sigma[T]/(M_all - len(chr_map[ block.chrom ]))
        elif block.sigma[T]>0: # we need to form the weights M_k - M_k,C
            VZ[:,upd_indx_1] += XDXZ[:,upd_indx_2]*block.sigma[T]/weights # elementwise

    # if debug: printLog("Updates: {0:.2f}.".format(time.perf_counter()-tic1), end='\n')
    if debug: printLog("Total: {0:.2f}".format( time.perf_counter()-tic0))
    return VZ

# end-of-file