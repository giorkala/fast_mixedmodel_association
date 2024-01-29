#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for MTMA:Fast, or the pseudo-loco approach

Created on Tue April 12th
@author: kalantzi
"""

import numpy as np
import pandas as pd
from pysnptools.snpreader import Bed
from scipy.sparse import diags
from scipy.stats import chi2
from scipy.stats import  linregress as LR
from multiprocessing.pool import ThreadPool as Pool
import time, os.path
from . import find_chromosomes, var_chunked, dot_chunked, convert_bed_to_hdf5, streaming_preparation_mt, find_noncausal, find_components_chunked, geno_blocks, printLog

def FMA_pseudo_loco( save_as, plinkfile, Traits, samples, annotation, VC, arg_uniform=False, arg_Ncal=100, cores=5, cg_max_iters=30, arg_debug=False, useHDF=None ):
    """
    Multi-Trait Mixed-model Association - the fast pseudo-loco version. 
    Step 1: Calculate genome-wide residuals  
    Step 2: Form LOCO residuals and calibrate accordingly.
    Note that N_calibr is NOT as before, as it controls the estimation of the calibration of the LOCO-based test statistics
    Should be followed by `FMA_JustTest` to get the test statistics.
    INPUT:  0) Traits: N x N_phen dataframe with phenotypes; should have a header
            1) VC: (N_comp + 1) x N_phen array with VC for each phenotype
            2) the rest are as before
    OUTPUT: 1) N_chr sets of residuals
            2) ones for calibrators as we don't need them anymore
            3) other info, e.g. number of iterations
    """
    print("="*56)
    # the next are variables to be used by each spawn process during the core calculations
    global N_phen, N_chr, N_comp, N_calibr, nrhs, N, M_all, batch_size
    global snp_on_disk, samples_to_use, bSizeA, bSizeB, hdf5, chr_map, block_info_all, debug, readings
    debug = arg_debug
    
    printLog("MTMA: Pseudo-LOCO + Many traits + Fast Streaming with Multi-Processing \n\n1. Initializations")
    if cores == '1':
        print("Will only process 1 block per genotype. Consider using more with `--nCores` for a speed-up.")
    else:
        print("Will run",cores,"computational processes in parallel.")
    time0 = time.perf_counter()
    snp_on_disk = Bed(plinkfile, count_A1=True)
    chr_map = find_chromosomes(plinkfile); 
    N, M_all = snp_on_disk.shape 
    N_chr = len(chr_map)
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
        Traits = Traits.loc[ Traits.index[samples] ] # or:
    # Remember, we dont change the phenotypes at all here
        
    print("\nGetting SNP mean/std info:") 
    col_means, col_vars = streaming_preparation_mt( snp_on_disk, int(M_all/500), Traits, samples, debug ) # df_LR, 
    
    bad_snps = np.where(col_means<=2e-4)[0] # so that MAF >=1e-4
    if len(bad_snps)>0: 
        print("\nWARNING: There are monomorphic/ultra-rare markers which will be excluded.")
        print(bad_snps[:5])
    
    N_phen = Traits.shape[1] # as we dont pass FID+IID
    N_comp = VC.shape[0]-1
    printLog("\n{0} samples, {2} chromosomes, {3} components, and {4} traits.".format(N, M_all, N_chr, N_comp, N_phen))
#    assert N_comp==N_chr, "The number of components is wrong"

    block_ID_all = find_components_chunked( annotation, chr_map, bad_snps, N_comp, bSizeB )
    if sum([ len(x) for x in block_ID_all.values()]) != M_all: # | ( int(sum(component_sizes))!=M_all*N_comp):
        print("\nWARNING: the number of annotated variants is different than the total number of SNPs!")
        print( sum([ len(x) for x in block_ID_all.values()]) ) #, sum(component_sizes) )
        M_all = sum([len(x) for x in block_ID_all.values()])
    
    if debug:
        print("Variance component estimates:")
        for T in range(N_phen):
            print(*VC[:,T]) # print all the values without brackets
    
    N_calibr = 0 # not really need but it's part of the code
    if arg_Ncal>0:
        print(arg_Ncal,"SNPs will be used per phenotype to calibrate the LOCO residuals.")
    else:
        print("WARNING: No calibration will be performed and test statistics might be inflated!")
        
    # filter out traits with unreasonably low Ve estimates
    good_traits = np.where( VC[-1,:]>=0.20 )[0] 
    if len(good_traits) < N_phen:
        print("\nWARNING:", *Traits.columns[np.where( VC[-1,:]<0.20 )[0]], "might fail to converge and should be excluded from the analysis!")

    if useHDF == None:
        # will create a new HDF5 file, the recommended way
        hdf5 = save_as+".fast.hdf5"
        convert_bed_to_hdf5(plinkfile, block_ID_all, samples, hdf5, chunk_size=1000)
    else:
        # use the given filename for the HDF5 and check if its there
        hdf5 = useHDF
        if not os.path.isfile( hdf5 ): # check if one exists already 
            print("Error: "+hdf5 + " was not found!")
            convert_bed_to_hdf5(plinkfile, block_ID_all, samples, hdf5, chunk_size=1000)
    
    # we'll use a dict to map each (k,C) ID to the corresponding "block_info" structure 
    block_info_all = {}
    left_end=0
    for block in block_ID_all: # block="(comp, chrom)"
        block_info_all[block] = block_info(block, block_ID_all[block], left_end, VC[block[0],:] )
        left_end += len( block_ID_all[block] )
        # TODO: is it likely to have a sigma << 0 ?
        block_info_all[block].means = col_means[ block_ID_all[block] ]
        block_info_all[block].weights = col_vars[ block_ID_all[block] ]**(-0.5) # the std of each SNP
        
    nrhs = Traits.shape[1] # no need for more systems in this setting
    print("\n{0} RHS in total are set for the linear systems.".format(nrhs))
    print("Initializations are complete. Duration for this step = {0:.2f} mins.".format( (time.perf_counter()-time0)/60 ) )

    ##################################
    ## Core computations start here ##
    print("\n2. Core calculations")
    time1 = time.perf_counter()
    readings = 0
    # Conjugate Gradients
    threshold = np.linalg.norm(Traits.to_numpy(), ord=2, axis=0)*5e-5
    CG_pack, norms, q = solve_conj_grad( Traits.to_numpy(), VC, threshold, cg_max_iters )

    print("CG is done; Total time for {1} systems and {2} iterations: {0:.2f} mins.\n".format( (time.perf_counter()-time1)/60, CG_pack.shape[1], q+1) )
    print("Total number of readings:",readings)
    ## Core calculations end here ##        
    ################################

    ####################################
    print("\n3. Calculation of LOCO residuals calibration")
    df = pd.DataFrame(CG_pack[:,:N_phen]); df.to_csv( save_as+".GW_residuals.gz", header=None, sep='\t', compression='gzip')
    # get the loco residuals
    residuals_all = calculate_loco_residuals( Traits, CG_pack, VC, save_as )
    print("Genome wide residuals are saved in "+save_as+".GW_residuals.gz")
    ##################################

    ###############################
    # get the calibration factors #
    calibrators = {}; 
    if arg_Ncal>0:
        subsize = N//2 if N<20000 else 10000
        print("\nGetting LR-adjusted calibration constants using {0} samples.".format(subsize))
        tic = time.perf_counter()
#        POOL = Pool(batch_size) # turns out MP is slower for this part
#        temp = POOL.starmap( get_loco_calibrators, ( (Traits.iloc[:,T].to_numpy(), T, residuals_all, arg_Ncal, 5000) for T in range(N_phen)) )
#        POOL.close (); POOL.join () # close; do we need this??      
        for T in range(N_phen):
#            calib, nr = temp[T]
            calib, nr = get_loco_calibrators( Traits.iloc[:,T].to_numpy(), T, residuals_all, arg_Ncal, subsize )
            print("{0}: Average number of markers used: {1:.1f}. Per-chrom calibrators:".format(Traits.columns[T],nr))
            print(" ".join(map("{0:.4f}".format,calib)))
            calibrators[ Traits.columns[T] ] = calib #np.mean(temp) 
        print("\nTime for calculating the calibration factors = {0:.2f} secs.".format( time.perf_counter()-tic ))
    else:
        print("\nWARNING: Using ones as calibration constants!")
        for T in range(N_phen):
            calibrators[ Traits.columns[T] ] = 1
 
    print("\nModel fitting is done! Total duration for {0} traits = {1:.2f} mins (plus trait preparation).".format( N_phen, (time.perf_counter()-time0)/60 ))
    print("="*45)
    
    # perform the testing part
    FMA_test_many_LD( plinkfile, residuals_all, calibrators, samples, save_as, verbose=True )
    return [residuals_all, calibrators, q]
  

def FMA_test_many_LD( bedfile, residuals_wrap, Traits, pheno_names, samples=None, saveas="FMA.sumstats", verbose=True ):
    """
    Test for association given the pseudo-LOCO residuals and calibrate wrt LD, for many traits. We need both the residuals and 
    the phenotypes as input because the calibration is performed here. The input could be just paths to files or arrays etc. 
    It both returns the dataframe, and saves it on disk (if asked).
    INPUT:
        bedfile: path to the hdf5 with genotypes
        residuals: dict of dataframes of the form NxN_pheno (could take 4+ GBs for N=500k and 22 chroms)
        Traits: numpy array of size N_eff x N_pheno
        samples: index of individuals to include in analysis; 
    TODO: samples should be different for each trait, according to missingness
    TODO: variant QC filters
    TODO: Expand for non-bed formats and dosages
    """
    tic0 = time.perf_counter()
    snp_on_disk = Bed(bedfile, count_A1=True)
    _, M_all = snp_on_disk.shape
    temp = pd.read_csv(bedfile+".bim", sep='\t', header=None) # we'll need this for SNP info
    chr_map = find_chromosomes(bedfile); # SNP indices per 
    header = True # for the csv files
    BS = 500 # batch-size per chrom
    N_batches = int(M_all/500)
    N_phen=len(pheno_names)

    if type(residuals_wrap)==str:
        # if a str is passed, we assume thats a path to the file and we load that
        residuals = {}
        for trait in pheno_names:
            residuals[trait] = pd.read_csv(".".join( [residuals_wrap,trait,"residuals.gz"]), sep='\t').iloc[:,2:].to_numpy()
    else: # for when an array is passed directly
        residuals = residuals_wrap

    if samples == None:
        # if no information is passed, we assume that all the available samples need to ba analysed
        N = residuals[0].shape[0]
        samples = range(N)
    else:
        N = len(samples)
    if verbose: print("\nFMA-Fast: Calculate SNP test statistics for {0} samples, using pre-computed residuals, for {1} traits.".format(N, N_phen) )
    assert len(residuals) == len(chr_map), "Error with the number of chromosomes!"
    assert residuals[0].shape[0] == N, "Shape of residuals is not as expected!"
    assert residuals[0].shape[1] == N_phen, "Shape of residuals is not as expected!"

    # initialise files with sum-stats; could just replace with a header but this way we might be able to spot a bug, if any
    for T in range(N_phen):
        if os.path.exists(".".join( [saveas,pheno_names[T],"sumstats.gz"])):
            os.remove(".".join( [saveas,pheno_names[T],"sumstats.gz"]))
            
    ##### load the M ld-scores and get the 50 quintiles #####
    Nbins=50
    df_snp = pd.read_csv(bedfile+".score.ld", sep='\s+')
    assert len(df_snp) == M_all, "Mismatch between M_all and ld_scores"
    bins = np.zeros(Nbins+1)
    for i in range(Nbins):
        bins[i] = np.percentile(df_snp.ldscore,2*i)
    bins[-1] = np.max(df_snp.ldscore)
    centers = [ (bins[i]+bins[i+1])/2 for i in range(Nbins) ]
    
    ##### get baseline LR test stats #####
    print("Reading the bed file for SNP info and LR-based statistics. This will take a few minutes as we will process",N_batches,"batches. Processed so far:")
    tic = time.perf_counter()
    varY = [Traits[:,T].dot(Traits[:,T]) for T in range(N_phen)] # should be ~equal to N-C anyway
    col_means = np.empty(M_all); col_vars = np.empty(M_all); col_nans = np.empty(M_all)
    stats_LR = np.zeros((M_all,N_phen))
    for i in range(0, N_batches+1):
        if i%10==0: print("B"+str(i), end=', ')
        index = range(i*BS, min((i+1)*BS,M_all))
        X = snp_on_disk[ samples,index].read( dtype='int8' ,  _require_float32_64=False ).val
        col_means[index] = np.mean(X, axis=0)
        col_vars[index] = var_chunked(X, blocks=4) 
        col_nans[index] = len(samples) - np.sum(np.isnan(X),axis=0)
        X0 = X - col_means[index] 
        xTy = X0.T.dot(Traits) # Mb x N_pheno
        for T in range(N_phen):
            stats_LR[index,T] = (N-2) / ( varY[T] *col_vars[index]*N /xTy[:,T]**2 - 1 )
    print("Done. Duration for this step = {0:.2f} mins".format( (time.perf_counter()-tic)/60 ))

    ##### get the uncalibrated test statistics #####
    print("\nGetting the FMA uncalibrated test-stats. Processing chromosome ",end='')
    tic = time.perf_counter()
    stats_ucFast = np.zeros((M_all,N_phen))
    betas = np.zeros((M_all,N_phen))
    for C,CHR in enumerate( chr_map ):
        print(C,end=', ')
        Mc = len(chr_map[CHR])
        N_batches=Mc//BS; # TODO: make this an option?
            
        for i in range(0, N_batches+1):
            index = chr_map[CHR][i*BS : min((i+1)*BS,Mc)]
            geno_matrix = snp_on_disk[ samples,index].read( dtype='int8' ,  _require_float32_64=False ).val
            # the next gives x * Vinv * y for any x, where x should be mean-centered
            numerators = geno_matrix.T.dot(residuals[C])
            numerators = numerators - np.diag(col_means[index]).dot( np.ones((len(index),1)).dot(np.sum(residuals[C], axis=0).reshape(1,N_phen))) 
           
            for T in range(N_phen):
                # the next is norm(x)^2 * norm(Vinv_y)^2:
                denominators = col_vars[index] * residuals[C][:,T].dot(residuals[C][:,T])
                stats_ucFast[index,T] = numerators[:,T]**2 / denominators
                betas[index,T] = numerators[:,T] / denominators # TODO: find a way to debias these
    print("Done. Duration for this step = {0:.2f} mins".format( (time.perf_counter()-tic)/60 ))
    
    sum_stats = pd.DataFrame(columns=["SNP", "CHR","BP","A1","A2","AFREQ","N","BETA","SE","CHISQ","P","CHISQ_LR"])
    sum_stats['SNP'] = temp[1]; sum_stats['CHR'] = temp[0]; 
    sum_stats['BP'] = temp[3]; sum_stats["AFREQ"] = col_means/2
    sum_stats['A1'] = temp[4]; sum_stats['A2'] = temp[5];
    sum_stats['N'] = col_nans
        
    for T in range(N_phen):
        ##### find the LD-based calibrators #####
        mean_chi2 = np.zeros((Nbins,2))
        for i in range(Nbins):
            indx = np.where( ( df_snp.ldscore>=bins[i] ) & ( df_snp.ldscore < bins[i+1]) )[0] 
            mean_chi2[i,0] = np.mean( stats_ucFast[indx,T] )
            mean_chi2[i,1] = np.mean( stats_LR[indx,T] )
            
        inflation = mean_chi2[:-1,0]/ mean_chi2[:-1,1]
        slope, intercept, r2, pval, se = LR(centers[:-1], inflation)
        print('\n'+pheno_names[T]+": slope={0:.4f}, intercept={1:.4f}, rvalue={2:.4f}, pvalue={3:.2e}".format(slope, intercept, r2, pval))
      
        ### save sumstats to disk ###
        sum_stats["CHISQ"] = stats_ucFast[:,T]/( df_snp.ldscore * slope + intercept)    
        sum_stats['BETA'] = betas[:,T]
        sum_stats['SE'] = np.abs(sum_stats['BETA']) / np.sqrt(sum_stats["CHISQ"])
        sum_stats['P'] = chi2.sf( sum_stats['CHISQ'] , df=1)
        sum_stats["CHISQ_LR"] = stats_LR[:,T]
        sum_stats.to_csv( ".".join( [saveas,pheno_names[T],"sumstats.gz"]), sep='\t', index=None, header=header, compression='gzip' ) 
        
        if verbose: 
            print("Mean LinReg: {0:.4f}  ({1} good SNPs)   lambdaGC: {2:.4f}".format( np.mean(stats_LR[:,T]), M_all, np.median(stats_LR[:,T])/chi2.ppf(0.5, df=1) ))
            print("Mean ucFMA: {0:.4f}  ({1} good SNPs)   lambdaGC: {2:.4f}".format( np.mean(stats_ucFast[:,T]), M_all, np.median(stats_ucFast[:,T])/chi2.ppf(0.5, df=1) ))
            print("Mean FMA  : {0:.4f}  ({1} good SNPs)   lambdaGC: {2:.4f}".format( np.mean(sum_stats["CHISQ"]), M_all, np.median(sum_stats["CHISQ"])/chi2.ppf(0.5, df=1) ))
    
    print("\nDuration for test-statistics = {0:.2f} mins".format( (time.perf_counter()-tic0)/60 ))
    
    return

#%% Support functions; only specific to streaming    
class block_info:
    def __init__(self, label, index, starting_from, vc):
        self.label = label
        self.index = np.arange(starting_from,starting_from+len(index) )
        self.size  = len(index) # this is for the math within XX'Z, not for the final weight
        self.sigma = vc
        self.compo = label[0]
        self.chrom = label[1]
        self.chunk = label[2]
        self.means = np.empty( self.size )
        self.weights = np.empty( self.size )
# TODO: add a function here that loads the corresponding block of genotypes?

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
        
        VZ = covar_matmat_noloco(Z, VC)
        
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
        printLog("CG iteration {0}: time={1:.2f}, mean norm: {2:.6f}. Block sizes: {3} , {4}.".format(q+1, time.perf_counter()-tic, np.nanmean(norms),  bSizeA, bSizeB ), flush=True)

    return solutions, norms, q
  
def covar_matmat_noloco( X, VC ):
    """
    Calculates the V*X, where V is formed according to VC, using multiprocessing.
    """
    global N_phen, N_chr, N_calibr, readings, block_info_all, batch_size

    VX = np.zeros(X.shape, dtype=np.float64)
    # add the environmental component
    M_1 = len(chr_map[0])
    for T in range(N_phen):
        VX[:,T] = VC[-1,T]*X[:,T]
#        upd_indx_tmp = np.arange(T*(1+N_calibr), (T+1)*(1+N_calibr)) #
        upd_indx_tmp = N_phen + np.arange(T*N_calibr, (T+1)*N_calibr)
       
        sigma_e_loco = 1 - ( M_all - M_1 ) * ( 1 - VC[-1,T] ) / M_all
        VX[:,upd_indx_tmp] = X[:,upd_indx_tmp] * sigma_e_loco
        
    for i in range(len(block_info_all)//batch_size +1):
        index = range(i*batch_size, min((i+1)*batch_size,len(block_info_all)))
        POOL = Pool(batch_size)
        # POOL = multiprocessing.Pool(cores)
        partial_VX = POOL.starmap( core_calc_batch, ( (b,X) for b in [ list(block_info_all.values())[x] for x in index]))
        # ideally each worker should write directly on `VX` but I can't find a way to do that
        POOL.close (); POOL.join () # close; do we need this??      
        readings += len(partial_VX)
        for x in partial_VX:  
            if x.shape[1] == N_phen:
                VX[:,:N_phen] += x
            else:
                VX += x
        
    return VX
  
def core_calc_batch(block, Z):
    """
    Calculate XX'Z, and adjust with the corresponding weight, where X is the standardized form of block G and Z is defined by the solver.
    This is the hardest part of the framework and its highly optimised to avoid unnecessary casting or intermediate arrays.
    Some hyperparameters are `bSizeA`, `bSizeB`, which control the sizes for block-wise calculations, 
    and `uniform` which controls what weights we'll multiply the GRMs with.
    """
    # 1. get the sub-GRM product - assuming there are SNPs for c_inf estimation (deprecated)
    if block.chrom > 0:
        XDXZ = get_XXtZ(block, Z)
    else:
        XDXZ = get_XXtZ(block, Z[:,:N_phen])
    # 2. adjust according to VC factors; each block.sigma is now a vector of values and each set of columns needs to be assigned to a different one
    # first we need to get M_k by looking at all the blocks of current component
#    M_1 = len(chr_map[0]) # TODO
    tic0 = time.perf_counter()
    for T in range(N_phen):
        # select which systems involve the current VCs: ONE residual + Ncalibr for the current pheno
#        indx_trait = np.union1d( T, N_phen + np.arange(T*N_calibr, (T+1)*N_calibr) )
#        if uniform: 
        XDXZ[:,T] *= block.sigma[T] / M_all
        
#        if block.chrom > 0:
#            indx_trait = N_phen + np.arange(T*N_calibr, (T+1)*N_calibr)
#            sigma_loco = block.sigma[T] * (M_all - M_1) / M_all
#            XDXZ[:,indx_trait] *= sigma_loco / (M_all - M_1) 
    #            else:
#                # "drop these columns"
#                indx_trait = N_phen + np.arange(T*N_calibr, (T+1)*N_calibr)
#                XDXZ[:,indx_trait] *= 0 # np.zeros( (N,N_calibr) )
#        elif block.sigma[T]>0: # we need to form the weights M_k - M_k,C
#            weights = np.repeat( block.size, 1+N_calibr) # OCO: just one batch
#            XDXZ[:,indx_trait] *= block.sigma[T]/weights # elementwise

    if debug: printLog("Total: {0:.2f}".format( time.perf_counter()-tic0))
    return XDXZ
  
def get_XXtZ(block, Z):
    """
    Calculate XX'Z where X is the standardized form of block G and Z is defined by the solver.
    Some hyperparameters are `bSizeA`, `bSizeB`, which control the sizes for block-wise calculations
    """
    global samples_to_use, bSizeA, bSizeB
    #     BS = 500* 2**17 // G.shape[0] # so as each chunk takes less than 500MB of RAM for casting to float -- could be similar to bSizeB
#    N_batches = N//bSizeB if N%bSizeB==0 else N//bSizeB+1  # params for N-based calculations
    
    Z_sum = np.sum(Z, axis=0).reshape(1,Z.shape[1])
    G = geno_blocks( block.index, samples_to_use, hdf5 ) # read from disk; `SNPs x samples` format
    
    tic1 = time.perf_counter()
    XZ = dot_chunked( G, Z, bSizeA ) # mat-mat ONE 
    if debug: printLog("MatMat1: {0:.2f}.".format(time.perf_counter()-tic1), end=' ')
    temp1 = diags(block.weights).dot(XZ) # D*X*Z
    del XZ # not needed anymore (new)
    
    temp2 = diags(block.weights).dot( block.means.reshape(block.size, 1).dot(Z_sum) )  # D \mu 1 U
    
    tic1 = time.perf_counter()
    XDXZ = dot_chunked( G.T, temp1 - temp2, bSizeB) # mat-mat TWO
    if debug: printLog("MatMat2: {0:.2f}.".format(time.perf_counter()-tic1), end=' ')
    
    XDXZ -= np.ones((G.shape[1], 1)) * block.means.T.dot(temp1)  ## TERM-2
    XDXZ += block.means.T.dot( diags(block.weights).dot(block.means.reshape(block.size, 1).dot(Z_sum)) )  ## TERM-4
    
    return XDXZ


def calculate_loco_residuals( Traits, Vinv_y_all, VC, prefix ):
    """Expand the genome-wide Vinv_y residuals to LOCO ones by adding the appropriate terms 
    This should be invoked once for all traits for a minimal set of genotype readings

    Args:
        Traits: NxP
        Vinv_y ([type]): [description]
        VC: (nCompo+1 x P)
        chrom_index ([type]): [description]

    Returns:
        dict of N_chr arrays of size `N x N_pheno` of LOCO-like residuals
    """
#    nComp = VC.shape[0]-1
    residuals_all = {};
    
    # 1a. get the BLUP estimates
    print("\nGetting the Best Linear Unbiased Estimates...", end=' ')
    tic = time.perf_counter()
    betas = np.zeros( (M_all, N_phen), dtype=np.float64 )
    for block in block_info_all.values():
        betas[block.index,:] = get_XtZ( block, Vinv_y_all ) # TODO: maybe use MP here?
    # 1b. adjust accordingly for sigma_g^2/M
    for T in range(N_phen):
        betas[:,T] *= (1-VC[-1,T])/M_all 
    
    print("Done. Duration: {0:.2f}".format( time.perf_counter()-tic))
    
    df = pd.DataFrame(betas); df.to_csv( prefix+".blues.gz", header=None, sep='\t', compression='gzip' )
    print("BLUEs are saved in "+prefix+".blues.gz")
    
    # 2 calculate pseudo-loco residuals
    print("Calculating LOCO-like residuals...", end=' ')
    tic = time.perf_counter()
    BLUP_all = {} # one entry per chromosome, for all phenotypes
    for C in range(N_chr):
        BLUP_all[C] = np.zeros( (N,N_phen), dtype=np.float64 )
        residuals_all[C] = np.copy(Traits) # initialise
    
    readings = 0; # TODO: maybe use MP here?
    for block in block_info_all.values():
        BLUP_all[block.chrom] += get_XZ( block, betas[block.index,:] )
        readings += len(block.index)
                
    for C1 in range(N_chr):
#        sigma_e_loco = 1 - ( 1 - VC[-1,:] ) * ( M_all - len ) / M_all
        for C2 in range(N_chr):
            if C2 != C1:
                residuals_all[C1] -= BLUP_all[C2]
        residuals_all[C1] /= VC[-1,:] # TODO: use loco estimates
    print("Done. Number of reads: {0}. Duration: {1:.2f}".format( readings, time.perf_counter()-tic))
    return residuals_all


def get_loco_calibrators( pheno, T, residuals_all, Ncalib = 50, subsize=5000 ):
    """
    A function to estimate the inflation in test statistics. We need to carefully select which SNPs to 
    consider as we 1) avoid over-correcting the causal variants, 2) also skip cases with really high chi2 
    as those will cause deflation, and 3) skip cases with chi2<1 (either) as the corresponding ratios 
    can have arbitrary values (thus high std).
    """
    calibr = np.zeros(N_chr)
    used = 0
    for c in range(N_chr):
        subset = np.random.permutation(N)[:subsize]
        Vinv_y = residuals_all[c][subset,T]
        calibr_snps, lr_test = find_noncausal( snp_on_disk, pheno[subset], subset, chr_map[c], Ncalib, 1, 3 ) #iloc[subsample]
#        calibr_snps, lr_test = find_noncausal( snp_on_disk, pheno[subse], range(N), chr_map[c], Ncalib, 1, 3 ) #iloc[subsample]
        chisqr_new = np.zeros(Ncalib)
        var0 = Vinv_y.dot(Vinv_y)
        for m,g in enumerate(calibr_snps.T):
            temp = g.T.dot(Vinv_y)**2
            chisqr_new[m] = (len(subset)-2)*temp /g.dot(g) /var0
        good = np.where( (chisqr_new>0.75) & (chisqr_new<4) )[0] # skip any outliers
        calibr[c] = ( np.mean( chisqr_new[good] / lr_test[good] ) + np.median( chisqr_new[good] / lr_test[good] ))/2
        used += len(good)
    return calibr, used/N_chr


def get_XtZ( block, Z ):
    """
    Calculate XZ where X is the genotypes that are var-adjusted of block (Mb x N) and Z is any array (N x P).
    """
    G = geno_blocks( block.index, samples_to_use, hdf5 ) # read from disk; `SNPs x samples` format
#    G = (G - np.ones((N,1)).dot( np.mean(G, axis=0).reshape(1,block.size))).dot(np.diag(1/np.std(G, axis=0)))
#    return G.T.dot(Z)
    XZ = diags( block.weights**2 ).dot( dot_chunked( G, Z, bSizeA ) ) 
    del G
    Z_sum = np.sum(Z, axis=0).reshape(1,Z.shape[1])
    XZ -= diags( block.weights**2 ).dot( block.means.reshape(block.size,1).dot(Z_sum))  
    return XZ
  
def get_XZ( block, Z ):
    """
    Calculate XZ where X is the standardized genotypes of block (N x Mb) and Z is any array (Mb x P). 
    """
#    temp = diags( block.weights ).dot(Z) # this is DZ: `MxP`
#    XZ = dot_chunked( geno_blocks( block.index, samples_to_use ).T, temp, bSizeB) # read and do the mat-mat with one shot
#    N_batches = N//bSizeB if N%bSizeB==0 else N//bSizeB+1
#    for i in range(N_batches):
#        index = range(i*bSizeB, min((i+1)*bSizeB,N))
#        XZ[index,:] -= np.ones((len(index),1))*block.means.T.dot( temp )
#    return XZ
    return dot_chunked( geno_blocks( block.index, samples_to_use, hdf5 ).T, Z, bSizeB)

  
#from scipy.optimize import minimize
#def get_loco_calibrators( Traits, T, gw_residual, sigmas, GGtY_all, subsize=25000 ):
#    """ OLD - no longer needed """
#    Ncalibr=100
#    # TODO: We could sub-sample here to reduce the running time
#    subsample = np.random.permutation(N)[:subsize]
#    pheno = Traits.iloc[subsample,T]
##    geno = snp_on_disk[ range(N), : ].read().val 
#    deltas = np.zeros(N_chr)
#    for C in range(N_chr):
#        calibr_snps, lr_test = find_noncausal(snp_on_disk, pheno, subsample, chr_map[C], Ncalibr)    
#        q1, q2, q3 = get_qs_one( sigmas, C )
#    
##        temp = Gs[:, chr_map[c]].T.dot(pheno)
##        temp0 = Gs[:, chr_map[c]].dot(temp)
#        GGtY = np.sum([GGtY_all[c][subsample,T] for c in range(N_chr)], axis=0)
#        def loss_lr_based(x, Ncalibr=100):
#            chisqr_new = np.zeros(Ncalibr)
#            a = gw_residual[subsample] + q1*pheno + q2*GGtY + q3*GGtY_all[C][subsample,T] * x
#            var0 = a.dot(a)
#            for m,g in enumerate(calibr_snps.T):
#                temp = g.T.dot(a)**2
#                chisqr_new[m] = (N-2)*temp /g.dot(g) /var0
#    #         return (1-np.median(chisqr_new/lr_test) )**2
#            return np.mean( (chisqr_new - lr_test)**2 )
#    
#        opt = minimize(loss_lr_based, 0)
#        deltas[C] = opt.x
#        # print(c, opt.x, opt.nit)  
#    return deltas
    
#def get_qs_one( sigmas, c ):
#    Mc = len(chr_map[c])
#    sigma_g = np.sum(sigmas)
#    sigma_g_loco = sum(sigmas[np.setxor1d(range(10),c)])
#    sigma_e = 1 - sigma_g
#    sigma_e_loco = 1 - sigma_g_loco
#    delta1 = sigma_g / sigma_e / M_all
#    delta2 = sigma_g_loco / sigma_e_loco / (M_all-Mc)
#    return 1/sigma_e_loco - 1/sigma_e, delta1/sigma_e - delta2/sigma_e_loco, delta2/sigma_e_loco
#    
#def get_qs_many( VC, c ):
#    """
#    Calculate the q1,q2,q3 coefficients for the LOCO approximation, based on "Eq.8"
#    VC needs to be (N_chr+1) x N_phen
#    """
#    assert VC.shape[0]==N_chr+1, "The number of components is wrong"
#    assert VC.shape[1]==N_phen,  "The number of phenotypes is wrong"
#    Mc = len(chr_map[c])
#    sigma_g = np.sum( VC[np.arange(N_chr),:], axis=0 ) # should have T values
#    sigma_g_loco = np.sum(VC[ np.setxor1d(range(10),c), :], axis=0)
#    sigma_e = 1 - sigma_g
#    sigma_e_loco = 1 - sigma_g_loco
#    delta1 = sigma_g / sigma_e / M_all
#    delta2 = sigma_g_loco / sigma_e_loco / (M_all-Mc)
#    return 1/sigma_e_loco - 1/sigma_e, delta1/sigma_e - delta2/sigma_e_loco, delta2/sigma_e_loco
    
## end-of-file