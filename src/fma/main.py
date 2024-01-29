# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:16:34 2020
@author: giork

TODO: update chrom_index to follow whatever is in the BIM file, instead of {0,1,..,Nchr}. To do that, we need to ensure that nothing breaks when we refer to chr-id from each block.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2
from scipy.sparse import diags 
from pysnptools.snpreader import Bed
from multiprocessing.pool import ThreadPool as Pool
import time, subprocess, os

class block_info:
    def __init__(self, label, index, starting_from, vc):
        self.label = label
        # self.index = index
        self.index = np.arange(starting_from,starting_from+len(index) )
        self.size  = len(index) # this is for the math within XXP, not for the final weight
        self.sigma = vc
        self.compo = label[0]
        self.chrom = label[1]
        self.means = np.empty( self.size )
        self.weights = np.empty( self.size )
        self.geno = None
   
def FMA_mem_version( plinkfile, Traits, samples, annotation, VC, arg_uniform=False, arg_Ncal=3, cores=4, cg_max_iters=50, arg_debug=False ):
    """
    Randomised Complex Trait Association - for multiple genetic components and many traits - MEMORY version
    Step 1: Calculate the LOCO residuals and the calibration factor.
    Should be followed by `FMA_JustTest` to get the test statistics. We will only get those for LR here. 
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
    global N_phen, N_chr, N_calibr, nrhs, nrhs_chrom, nrhs_chrom_pheno, N, M_all
    global samples_to_use, P, AP, bSizeA, bSizeB, uniform, block_ID_all, chr_map, debug
    debug = arg_debug

    printLog("FMA: Multiple Components + Many traits + Genotypes in memory \n\n1. Initializations")
    if cores == '1':
        print("Will only process 1 block per genotype. Consider using more with `--nCores` for a speed-up.")
    else:
        print("Will run",cores,"computational processes in parallel.")
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
    
    samples_to_use = samples
    
    N_phen = Traits.shape[1] # as we dont pass FID+IID
    N_comp = VC.shape[0]-1
    block_ID_all = find_components( annotation, N_comp, chr_map )

    if sum([ len(x) for x in block_ID_all.values()]) != M_all: # | ( int(sum(component_sizes))!=M_all*N_comp):
        print("\nWARNING: the number of annotated variants is different than the total number of SNPs!")
        print( sum([ len(x) for x in block_ID_all.values()]) ) #, sum(component_sizes) )
    
    if Traits.shape[0] != len(samples): 
        print("\nWARNING: number of phenotyped samples doesn't match the fam size!")
        Traits = Traits[samples]
    # Remember, we dont change the phenotypes at all here
    
    printLog("{0} samples, {1} variants, {2} chromosomes, {3} components, and {4} traits.".format(N, M_all, N_chr, N_comp, N_phen))
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
    
    # filter out traits with unreasonably low Ve estimates
    good_traits = np.where( VC[-1,:]>=0.20 )[0] 
    if len(good_traits) < N_phen:
        print("\nWARNING:", *Traits.columns[np.where( VC[-1,:]<0.20 )[0]], "might fail to converge and should be excluded from the analysis!")

    ###################################
    ## Genotypes will be loaded here ##
    print("Loading genotypes and calculating LINREG statistics...")
    # col_means = np.empty(M_all); col_vars = np.empty(M_all)
    df_LR = np.zeros( (M_all, Traits.shape[1])) # this will be of size M x Nt and will have proper SNP indexing
    
    # we'll use a dict to map each (k,C) ID to the corresponding "block_info" structure 
    block_info_all = {}
    right=0
    for block in block_ID_all: 
        # NOTATION: block="(comp, chrom)", block_ID_all[block]=index. 
        # We'll stop using this dict after the initialisation of `block_info_all`
        # 1. initialise class object
        block_info_all[block] = block_info(block, block_ID_all[block], right, VC[block[0],:] )
        # 2. load the genotypes from the bed file
        genotypes = snp_on_disk[ samples, block_info_all[block].index ].read( dtype='int8' ,  _require_float32_64=False ).val
        right += len( block_ID_all[block] )
        # 3. get means and weights for geno-standardization
        # col_means[block_ID_all[block]] = 
        # col_vars[block_ID_all[block]] = var_chunked(genotypes, blocks=4) 
        
        block_info_all[block].means = np.mean(genotypes, axis=0)
        try:
            block_info_all[block].weights = var_chunked(genotypes, blocks=4)**(-1)
        except:
            print("\nWARNING: There are monomorphic markers in block", block)
            
        # 4. Get LR statistics
        G0 = genotypes - block_info_all[block].means 
        for T in range(Traits.shape[1]):
            # test in batches and keep only the CHISQ statitics (no covariates)
            df_LR[ block_info_all[block].index, T ] = MyLinRegr( G0, Traits.iloc[:,T], W=None)["CHISQ"]
        # 5. compress the genotypes - this involves the creation of a new m*N*2 array of int8, so we'll do it in batches
        block_info_all[block].geno = geno_compress_efficient(genotypes.T, N_batches=5)
        # thats a very important step in the framework: the creation of an 1d array of compressed 8bit chunks of genotypes
        if debug: print(block, end=', ')
        
    # select all the non_causal variants that will be used for calibration
    RHS = np.zeros((N, N_phen*(N_chr + N_calibr*N_chr))) # for all the systems we will solve Nt*Nc*(1+Ncalibr)
    # example for N_calibr=2:
    # [ [y1, x1;1, x1;2], [y1, x2;1, x2;2], ..., [y1, x22;1, x22;2]] and repeat for y2, y3, ... yT
    col=0
    for T in range(N_phen):
        for C,CHR in enumerate(chr_map):
            RHS[:, col] = Traits.iloc[:,T]; col +=1
            indx = np.random.permutation( np.where( df_LR[ chr_map[CHR], T ] < 5 )[0] )[:N_calibr]
            # indx runs in {0,1,...,M_CHR} range
            for v in range(N_calibr):
                G = snp_on_disk[ samples, chr_map[CHR][indx[v]] ].read().val
                RHS[:, col] = (G - np.mean(G)).reshape(-1)
                col += 1
                
    print("Initializations are complete. Duration for this step = {0:.2f} mins.".format( (time.perf_counter()-time0)/60 ) )
    
    ##################################
    ## Core computations start here ##
    print("\n2. Core calculations")
    time1 = time.perf_counter()
    
    # Initialization
    max_iters = cg_max_iters; tol=5e-5
    CG_pack = np.zeros(RHS.shape, dtype=np.float64) # this will converge to the solution
    nrhs  = RHS.shape[1]
    r2old = np.zeros((nrhs,1), RHS.dtype)
    alpha = np.zeros((1,nrhs), RHS.dtype) # for elementwise
    beta  = np.zeros((1,nrhs), RHS.dtype) # for elementwise
    residuals = RHS.copy() # initialization, R(0) = B-A*X
    P = RHS.copy() # the "search directions"
    threshold = np.linalg.norm(RHS, ord=2, axis=0)*tol

    nrhs_chrom = N_phen*(N_chr-1)*(1+N_calibr) # the total Nupdates per chromosome
    nrhs_chrom_pheno = (N_chr-1)*(1+N_calibr) # the Ncolumns to update for one trait+one chrom
    readings = 0
    
    for q in range(max_iters): # or until convergence
        
        tic = time.perf_counter()
        
        AP = np.zeros(residuals.shape, dtype=np.float64)
        for T in range(N_phen):
            upd_indx_tmp = np.arange( T*N_chr*(N_calibr+1), (T+1)*N_chr*(N_calibr+1))
            AP[:,upd_indx_tmp] = VC[-1,T]*P[:,upd_indx_tmp] # the environmental component, shared across all cases
            
        for i in range(len(block_info_all)//batch_size):
            index = range(i*batch_size, min((i+1)*batch_size,len(block_info_all)))
            POOL = Pool(cores)
            # POOL = multiprocessing.Pool(cores)
            partial_AP = POOL.map( core_calc_mp, (b for b in [ list(block_info_all.values())[x] for x in index]))
            # ideally each worker should write directly on `AP` but I can't find a way to do that, thus I always keep `batch_size` matrices in 
            # the `partial_AP` list
            POOL.close (); POOL.join () # close; do we need this??      
            readings += len(partial_AP)
            for x in partial_AP:    
                AP += x
            del partial_AP
        
        # check the residuals and update w.r.t CG
        for j in range(nrhs):
            r2old[j] = residuals[:,j].dot( residuals[:,j] )
            alpha[0,j] = r2old[j] / P[:,j].dot( AP[:,j] ) 
        CG_pack +=  alpha * P # update the solutions; elementwise
        residuals -= alpha * AP 
        norms = np.linalg.norm(residuals, ord=2, axis=0)
        if (norms <= threshold).all(): # TODO: avoid the extra calculations in some cases (more than necessary)
            break # is this the proper condition? we end up with more iterations otherwise
        for j in range(nrhs):
            beta[0,j] = residuals[:,j].dot(residuals[:,j].T) / r2old[j]
        P = residuals + beta * P # update P; elementwise
        # P *= beta; P += residuals # is this better?
        # BOLT prints: iter 11:  time=18.29  rNorms/orig: (7e-07,0.0005)  res2s: 13277.4..13999.5
        printLog("CG iteration {0}: time={1:.2f}, mean norm: {2:.6f}. Block sizes: {3} , {4}.".format(q+1, time.perf_counter()-tic, np.nanmean(norms),  bSizeA, bSizeB ), flush=True)
    
    del residuals, P, AP
                
    ## Core calculations end here ##
    ################################
    time_CG = time.perf_counter()-time1
    print("CG is done; Total time for {1} systems and {2} iterations: {0:.2f} mins.\n".format( time_CG/60, CG_pack.shape[1], q+1) )
    print("Total number of readings:",readings)
    
    #################################
    ## Get the calibration factors ##
    col=0; 
    residuals_all = {}; calibrators = {};
    # for the rest, work in a trait-based logic
    for T in range(N_phen):
        # we should ideally go over a dictionary of converged cases which has the form "trait_index:trait_name"
        # so we need to be careful and skip non-converged cases accordingly
        # WARNING: if CG did not converge for a trait we dont calculate anything further
        print("\nWorking for {0} with total h2={1:.3f}".format(Traits.columns[T], 1-VC[-1,T]))

        Vinv_y_loco = np.empty((N,N_chr))   
        prospectiveStat = []; uncalibratedRetrospectiveStat = [] ## as in bolt  
        gamma = []
        # unpack the results from CG; these are ordered as [chr1-res, chr1-x1, ..., chr2-res, chr2-x1,..., chr22-x3]
        col = T*(N_chr + N_calibr*N_chr)
        for C in range(N_chr):
            Vinv_y_loco[:,C] = CG_pack[:,col]; col += 1
            for i in range(N_calibr):
                Xm = RHS[:, col] # get the original SNP
                temp = (Xm.T.dot(Vinv_y_loco[:,C]))**2
                # prospective = N * (x^T V^-1 y)^2 / (x^T V^-1 x * y^T V^-1 y)
                prospectiveStat.append( N*temp/ Xm.T.dot(CG_pack[:,col]) /np.dot(Traits.iloc[:,T], Vinv_y_loco[:,C]) )
                # TODO: replace N with N-C, adjusting for the number of covariates
                # prospectiveStat.append( temp/Xm.T.dot(CG_pack[:,col]) ) # this is probably wrong but similar to the original formula
                uncalibratedRetrospectiveStat.append( N*temp/Xm.T.dot(Xm)/Vinv_y_loco[:,C].dot(Vinv_y_loco[:,C]) ) # norm(Xm)^2 * norm(VinvY)^2
                gamma.append( Xm.T.dot(CG_pack[:,col])/Xm.dot(Xm) )
                col += 1
        print("Grammar-gamma approximation = {0:.4f} ({1:.4f})".format( np.mean(gamma), np.std(gamma)) )
    
        calibration = (sum(uncalibratedRetrospectiveStat) / sum(prospectiveStat)).item()    
        temp = np.array(uncalibratedRetrospectiveStat)/np.array(prospectiveStat)
        print("AvgPro: {0:.3f} AvgRetro: {1:.3f} Calibration: {2:.3f} ({3:.3f}) ({4} SNPs).".format(np.mean(prospectiveStat), np.mean(uncalibratedRetrospectiveStat), calibration, np.std(temp), len(prospectiveStat)))
        print("Ratio of medians: {0:.4f} | Median of ratios: {1:.4f}".format(  np.median(uncalibratedRetrospectiveStat)/np.median(prospectiveStat), np.median(temp))   )
        print("Mean, median and std of ratios = {0:.6f}, {1:.6f}, {2:.6f}".format(np.mean(temp), np.median(temp), np.std(temp) ))
        
        calibrators  [ Traits.columns[T] ] = calibration
        residuals_all[ Traits.columns[T] ] = Vinv_y_loco
        if norms[T*N_chr*(1+N_calibr)]>threshold[T*N_chr*(1+N_calibr)]:
            print("WARNING: CG has not converged for this trait!")
            calibrators[ Traits.columns[T] ] = np.nan
            
    print("\nModel fitting is done! Total duration for {0} traits = {1:.2f} mins (plus trait preparation).".format( N_phen, (time.perf_counter()-time0)/60 ))
    print("="*45)
    return [residuals_all, calibrators, time_CG, q]
    # return (time.perf_counter()-time0)/60


def FMA_test_many( bedfile, residuals_wrap, calibrators, samples=None, saveas="FMA.sumstats", verbose=True ):
    """
    Test for association given the LOCO residuals and calibration factors, for many traits, with fast streaming.
    The input could be just paths to files or arrays etc. It both returns the dataframe, and saves it on disk (if asked). 
    There shouldnt be any problems with trait indexing, as long as the residuals and calibrators are aligned.
    INPUT:
        hdf5: path to the hdf5 with genotypes
        residuals: dict of dataframes of the form NxN_pheno (could take 4+ GBs for N=500k and 22 chroms)
        calibrators: dict of the form "phenoname" -> "calibr-factor"
        samples: index of individuals to include in analysis; 
    TODO: samples should be different for each trait, according to missingness
    TODO: Expand for non-bed formats and dosages
    TODO: Run LR as well? We need to load the phenotypes or pass df_LR from the main function
    """
    time0 = time.perf_counter()
    snp_on_disk = Bed(bedfile, count_A1=True)
    temp = pd.read_csv(bedfile+".bim", sep='\t', header=None) # we'll need this for SNP info
    chr_map = find_chromosomes(bedfile); # SNP indices per 
    header = True # for the csv files
    BS = 500 # batch-size per chrom
    pheno_names = list(calibrators.keys())
    N_phen=len(calibrators)
    if verbose: chisqr_all = np.zeros((snp_on_disk.shape[1],N_phen)) # just in order to print the average/median

    if type(residuals_wrap)==str:
        # if a str is passed, we assume thats a path to the file and we load that
        residuals = {}
        for C in chr_map.keys():
            residuals[C] = pd.read_csv(".".join( [residuals_wrap,"loco",str(C),"residuals.gz"]), sep='\t').iloc[:,2:].to_numpy()
    else: # for when an array is passed directly
        residuals = residuals_wrap

    if samples is None:
        # if no information is passed, we assume that all the available samples need to ba analysed
        N = residuals[chr_map[0]].shape[0]
        samples = range(N)
    else:
        N = len(samples)
    if verbose: print("\nFMA: Calculate SNP test statistics for {0} samples, using pre-computed residuals, for {1} traits.".format(N, N_phen) )
    assert len(residuals) == len(chr_map), "Error with the number of chromosomes!"
    assert residuals[0].shape[0] == N, "Shape of residuals is not as expected!"
    assert residuals[0].shape[1] == N_phen, "Shape of residuals is not as expected!"

    # initialise files with sum-stats; could just replace with a header but this way we might be able to spot a bug, if any
    for T in range(N_phen):
        if os.path.exists(".".join( [saveas,pheno_names[T],"sumstats.gz"])):
            os.remove(".".join( [saveas,pheno_names[T],"sumstats.gz"]))
    
    print("Processing chromosome ",end='')
    # TODO: use Multiprocessing here too
    for C,CHR in enumerate( chr_map ):
        print(CHR,end=', ')
        Mc = len(chr_map[CHR])
        N_batches=Mc//BS; # TODO: make this an option?
            
        for i in range(0, N_batches+1):
            index = chr_map[CHR][i*BS : min((i+1)*BS,Mc)]
            
            geno_matrix = snp_on_disk[ samples,index].read( dtype='int8' ,  _require_float32_64=False ).val
            col_means = np.mean(geno_matrix, axis=0)
            col_vars = var_chunked(geno_matrix, blocks=10) 
            col_nans = len(samples) - np.sum(np.isnan(geno_matrix),axis=0)
            # the next gives x * Vinv * y for any x, where x should be mean-centered
            numerators = geno_matrix.T.dot(residuals[C])
            numerators = numerators - np.diag(col_means).dot( np.ones((len(index),1)).dot(np.sum(residuals[C], axis=0).reshape(1,N_phen))) 
           
            for T in range(N_phen):
                if np.ndim(calibrators[pheno_names[T]]) > 0:
                    # this corresponds to the pseudo-loco approach where we have different factors per chrom
                    calibrator = calibrators[pheno_names[T]][C]
                else:
                    calibrator = calibrators[pheno_names[T]]
                    
                sum_stats = pd.DataFrame(columns=["SNP", "CHR","BP","A1","A2","AFREQ","N","BETA","SE","CHISQ","P"]); #"CHISQ_LR",
                sum_stats['SNP'] = temp[1][index]; sum_stats['CHR'] = temp[0][index]; 
                sum_stats['BP'] = temp[3][index]; sum_stats["AFREQ"] = col_means/2
                sum_stats['A1'] = temp[4][index]; sum_stats['A2'] = temp[5][index];
                sum_stats['N'] = col_nans
                # the next is norm(x)^2 * norm(Vinv_y)^2:
                denominators = col_vars * residuals[C][:,T].dot(residuals[C][:,T])
                sum_stats["CHISQ"] = numerators[:,T]**2 / denominators / calibrator
                sum_stats['BETA'] = numerators[:,T] / denominators / calibrator  # as in Eq27 of PR Loh 2015
                sum_stats['SE'] = np.abs(sum_stats['BETA']) / np.sqrt(sum_stats["CHISQ"])
                sum_stats['P'] = chi2.sf( sum_stats['CHISQ'] , df=1)
                #  write on disk asynchronously; we write the header only the first time
                sum_stats.to_csv( ".".join( [saveas,pheno_names[T],"sumstats.gz"]), sep='\t', index=None, header=header, compression='gzip', mode='a') 
                if verbose: chisqr_all[index,T] = sum_stats["CHISQ"]
            header = False # dont write the header again
            
    print("Duration for test-statistics = {0:.2f} mins".format( (time.perf_counter()-time0)/60 ))
    if verbose: 
        for T in range(N_phen):
            print(pheno_names[T])
            print("Mean FMA  : {0:.4f}  ({1} good SNPs)   lambdaGC: {2:.4f}".format( np.mean(chisqr_all[:,T]), chisqr_all.shape[0], np.median(chisqr_all[:,T])/chi2.ppf(0.5, df=1) ))
    return
#%% Functions specific for compressing and handling compressed genotypes

def convert_to_haploid(X):
    """ takes MxN {0,1,2} array and returnd Mx2N {0,1}."""
    temp = np.zeros( (X.shape[0],2*X.shape[1]), dtype=np.uint8 )
    temp[:, range(0,2*X.shape[1],2)] = np.array( X>=1, dtype=int)
    temp[:, range(1,2*X.shape[1]+1,2)] = np.array( X>=2, dtype=int)
    return temp

def geno_compress_efficient(G, N_batches=10):
    """Start from a NxM diploid matrix G, take chunks of it, convert to haploid and return the 1d array of compressed values."""
    M, N = G.shape
    compressed = np.zeros( M*N//4, dtype=np.uint8 )
    
    batch_size = M//N_batches; 
    taken = 0
    for i in range(N_batches+1):
        index = np.arange(i*batch_size, min((i+1)*batch_size,M)) 
        # if debug: print(index)
        compressed[taken: taken+len(index)*N//4] = np.packbits( convert_to_haploid(G[index,:]) )
        taken += len(index)*N//4
    return compressed

def compressed_matmat(Z, U, m):
    """Gets the product X.dot(u) where X is the decompressed form of Z and u a vector; X is assumed MxN.
    Z is an 1-D array corresponding to m rows."""
    # TODO: is this always efficient?
    geno = np.unpackbits(Z).reshape(m,2*U.shape[0])
    return geno.dot(U.repeat(2, axis=0))

def compressed_matmat_transp(Z, U, N, m):
    """Gets the product X.dot(u) where X is the decompressed form of Z and u a vector. 
    Z is an 1-D array corresponding to m rows."""
    temp = (U.T.dot( np.unpackbits(Z).reshape(m,2*N) )).T
    return temp[range(0,2*N,2), :] + temp[range(1,2*N+1,2),:]

# the next two are not needed but good to have here
def compressed_matvec_simple(Z, u, m):
    """Gets the product X.dot(u) where X is the decompressed form of Z and u a vector. 
    Z is an 1-D array corresponding to m rows."""
    geno = np.unpackbits(Z).reshape(m,2*len(u))
    return geno.dot(u.repeat(2))

def compressed_matvec(Z, u, N_batches=20):
    """Gets the product X.dot(u) where X is the decompressed form of Z and u a vector. 
    Breaks Z accordingly to avoid memory bursts."""
    N = len(u)
    M = len(Z) // (N//4)
    batch_size = M//N_batches; 
    matvec = np.zeros(M,)
    
    for i in range(N_batches+1):
        index1 = np.arange( i*batch_size, min((i+1)*batch_size, M) ) # goes from 0,1,...,M
        index2 = np.arange( i*batch_size*N//4, min((i+1)*batch_size*N//4, M*N//4)) 
        matvec[index1] = compressed_matvec_simple(Z[index2], u, len(index1))
    return matvec

#%% The Great Set of Core Calculations
def core_calc_mp(block): #, bSizeA, bSizeB, uniform):
    global N_phen, N_chr, N_calibr, nrhs, nrhs_chrom, nrhs_chrom_pheno, N, M_all
    global samples_to_use, bSizeA, bSizeB, uniform
    if debug: printLog("hi", end=' ')

    # find which columns/sums this block needs to be included (all besides those of block.chrom):
    update_index = [True]*nrhs # # this includes all the traits
    for T in range(N_phen):
        for i in np.arange( block.chrom*(N_calibr+1), block.chrom*(N_calibr+1)+N_calibr + 1):
            update_index[ i + T*N_chr*(1+N_calibr) ] = False
    # if debug: print(update_index)
    
    ### calculate XX'P for standardized X ###
    P_sum = np.sum(P[:,update_index], axis=0).reshape(1,nrhs_chrom)
    # if debug: time.sleep(2) 
    tic0 = time.perf_counter()
#     BS = 500* 2**17 // G.shape[0] # so as each chunk takes less than 500MB of RAM for casting to float -- could be similar to bSizeB
    BS=bSizeB
    N_batches = N//bSizeB # params for N-based calculations
    # if debug: time.sleep(2)
    tic1 = time.perf_counter()
    G = np.unpackbits(block.geno).reshape(block.size,2*N)
    G = G[:, ::2] + G[:, 1::2]
    if debug: printLog("Getting geno: {0:.2f}.".format(time.perf_counter()-tic1), end=' ')
    # 1st dot product with genotypes; G is supposed to be MxN
    # XP = compressed_matmat( block.geno, P[:,update_index], block.size)
    XP = dot_chunked( G, P[:,update_index], bSizeA ) #int(np.sqrt(G.shape[1])) 
    if debug: printLog("MatMat1: {0:.2f}.".format(time.perf_counter()-tic1), end=' ')
    temp = diags(block.weights).dot(XP) # D*X*P
    del XP # not needed anymore (new)
    # if debug: time.sleep(1)
    tic1 = time.perf_counter()
    # if debug: print(G.shape, G.nbytes/1024**3)
    # 2nd dot product with genotypes
    XDXP = dot_chunked( G.T, temp, bSizeB) # int(100*np.log(N)) ) #
    # XDXP = compressed_matmat_transp(block.geno, temp, N, block.size)
    if debug: printLog("MatMat2: {0:.2f}.".format(time.perf_counter()-tic1), end=' ')
    # if debug: time.sleep(1)
    # the next produces a small spike in RAM
    # XDXP -= np.ones((N,1))*block.means.T.dot( temp ) # term-2; we use temp instead of diags(block.weights).dot(XP)
    for i in range(N_batches):
        index = range(i*BS, min((i+1)*BS,N))
        XDXP[index,:] -= np.ones((BS,1))*block.means.T.dot( temp )
    
    # prepare and add subtract the third term
    temp = diags( block.weights ).dot( block.means.reshape(block.size,1).dot(P_sum)) # D*mu*1*P
    # if debug: time.sleep(1)
    tic1 = time.perf_counter()
    # 3rd dot product with genotypes
    # dot_chunked_sub( G.T, temp, XDXP, G.shape[1]//1000)  
    # XDXP += compressed_matmat_transp(block.geno, temp, N, block.size)
    for i in range(0, N_batches+1):
        index = range(i*BS, min((i+1)*BS,N))
        XDXP[index, :] -= G.T[index,:].dot(temp)
        
    # if debug: print(XDXP.shape, XDXP.nbytes/1024**3)
    # if debug: time.sleep(1)
    if debug: printLog("MatMat3: {0:.2f}.".format(time.perf_counter()-tic1), end=' ')
        
    # the direct assignment produces a spike in memory, so we do it in two steps
#     XDXP += np.ones((N,nrhs_chrom))*block.means.T.dot( diags(block.weights).dot(block.means.reshape(block.size,1).dot(P_sum))) # part of term-4
    temp = block.means.T.dot( diags(block.weights).dot(block.means.reshape(block.size,1).dot(P_sum)))
    for i in range(N_batches):
        index = range(i*BS, min((i+1)*BS,N))
        XDXP[index,:] += temp

    # if debug: time.sleep(2)
    # now form the LOCO sums ("A*P") and complete the CG iteration
    tic1 = time.perf_counter()
    # each block.sigma is now a vector of values and each set of columns needs to be assigned to a different one
    # first we need to get M_k by looking at all the blocks of current component
    weights = np.repeat( sum( [len(block_ID_all[(block.compo, c)]) for c in range(N_chr) ] ), nrhs_chrom_pheno)
    # then for each chrom, update the corresponding weights accordingly
    chroms = set(range(N_chr)).difference([block.chrom])
    for i,C in enumerate(chroms):
        weights[np.arange( i*(N_calibr+1), i*(N_calibr+1)+N_calibr + 1)] -= len(block_ID_all[(block.compo, C)]) # assign the same number multiple times
    
    AP = np.zeros( (N,nrhs), dtype=np.float64)
    for T in range(N_phen):
        # all but block.chrom for trait T;
        indx_trait = np.arange( T*N_chr*(1+N_calibr), (T+1)*N_chr*(1+N_calibr))
        indx_chrom = np.arange( T*N_chr*(1+N_calibr) + block.chrom*(N_calibr+1), T*N_chr*(1+N_calibr) + (block.chrom+1)*(N_calibr+1))
        upd_indx_1 = np.setxor1d(indx_trait, indx_chrom)
        upd_indx_2 = np.arange( T*nrhs_chrom_pheno, (T+1)*nrhs_chrom_pheno )
        if uniform: 
            AP[:,upd_indx_1] += XDXP[:,upd_indx_2]*block.sigma[T]/(M_all - len(chr_map["chr"+str(block.chrom+1)]))
        elif block.sigma[T]>0: # we need to form the weights M_k - M_k,C
            AP[:,upd_indx_1] += XDXP[:,upd_indx_2]*block.sigma[T]/weights # elementwise

    # if debug: printLog("Updates: {0:.2f}.".format(time.perf_counter()-tic1), end='\n')
    if debug: printLog("Total: {0:.2f}".format( time.perf_counter()-tic0))
    return AP

#%% More functions-utilities

def printLog(*args, **kwargs):
    print(*args, **kwargs)
    with open('fma.debug','a') as file:
        print(*args, **kwargs, file=file)
        
def dot_chunked(A, B, BS ): #10
    """
    Perform A.dot(B) in batches to avoid memory overhead,
    avoiding casting the whole A to B.dtype.
    """
    N_batches = A.shape[0]//BS
    
    try:
        C = np.zeros((A.shape[0], B.shape[1]), dtype=B.dtype)
        for i in range(0, N_batches+1):
            index = range(i*BS, min((i+1)*BS,A.shape[0]))
            C[index, :] = A[index,:].dot(B)
    except:
        C = np.zeros((A.shape[0],), dtype=B.dtype)
        for i in range(0, N_batches+1):
            index = range(i*BS, min((i+1)*BS,A.shape[0]))
            C[index] = A[index,:].dot(B)       
    return C

def MyLinRegr(X, Y, W=None):
    """
    DIY Linear regression with covariates and low memory footprint.
    X should be NxM, for sufficiently large M (e.g. one chromosome), and mean centered
    Y should also be mean centered.
    W needs to be a NxC array with covariates including ones (the constant).
    Returns a DataFrame with estimated effect sizes, Chi-Sqr statistics, and p-values
    """
    if W is not None:
        # perform Linear Regression with covariates
        N, M = X.shape
        K = np.linalg.inv(W.T.dot(W)) # this is tiny
        y_hat =  Y - W.dot( K.dot( W.T.dot(Y) ) ) #P.dot(Trait)
        var_y = Y.dot(y_hat) # this is yTy
        temp = W.T.dot(X) # this is CxM (thus small)
        var_X = [ X[:,v].dot(X[:,v]) - temp[:,v].dot(K.dot(temp[:,v])) for v in range(M)] 
        sumstats = pd.DataFrame()
        numerators = X.transpose().dot( y_hat ) # this gives us xTy
        sumstats['BETA'] = numerators/ var_X
        sumstats['CHISQ'] = [(N-2)/( var_y*var_X[v]/numerators[v]**2 - 1 ) for v in range(M)]
        sumstats['P'] = chi2.sf( sumstats['CHISQ'] , df=1)
    else:
        sumstats = MyLinRegr_nocov(X, Y)
    return sumstats

def MyLinRegr_nocov(G, Trait):
    """DIY Linear regression without covariates (old; just for reference)"""
    N, M = G.shape
    sumstats = pd.DataFrame()
    numerators = G.transpose().dot(Trait) # this gives us xTy
    var_y = Trait.dot(Trait) # this is yTy
    var_x_all = [ G[:,v].dot(G[:,v]) for v in range(M)]
    sumstats['BETA'] = numerators/var_x_all
    sumstats['CHISQ'] = [(N-2)/( var_y*var_x_all[v]/numerators[v]**2 - 1 ) for v in range(M)]
    sumstats['P'] = chi2.sf( sumstats['CHISQ'] , df=1)
#     sumstats['Test_Stat'] = [ np.sqrt(N-2)*numerators[v]/np.sqrt(var_y*G[:,v].dot(G[:,v]) - numerators[v]**2) for v in range(M)]
#     sumstats['P'] = 2*t.sf(np.abs(sumstats['Test_Stat']), N-2)
    return sumstats

def find_chromosomes( bedfile ):
    """ 
    This function creates an index of SNP indices for each chromosome, from a given bim file.
    NOTE: we replace the bim chrom index (e.g. 1,2,5,22) to a pythonic one (e.g. 0,1,2,3).
    """
    df = pd.read_csv(bedfile+".bim", sep='\t', header=None)
    chrom_map = {}; start=0
    chroms = list(set(df[0])) # as in the bim file, should make sure this is "X" and not "chrX"
    # a plausible issue is that the above returns an unsorted list (e.g. [16,17,15]) and that will be devastating
    for i,C in enumerate(chroms):
        start = np.where( df[0]==C )[0][0] # index of first variant in this chrom
        end = np.where( df[0]==C )[0][-1]+1 # index of last variant in this chrom
        chrom_map[ i ] = np.array(range(start,end))
        #chrom_map[ "chr"+str(C) ] = range(start,end)
        start = end
    return chrom_map
  
def find_components( annotation, N_comp, chr_map ):
    """This function creates an index of chrom-wise SNP indices for each component, from a given annot file."""
    df = pd.read_csv(annotation, sep=' ', header=None)
    assert df.shape[1] == N_comp, "Annotation file and VC size mismatch!"
    class_map = {}; 
    print("Component sizes: ", end='')
    for k in range(N_comp):
        temp = list(np.where( df[k]==1 )[0])
        total = 0
        for C,CHR in enumerate(chr_map):
            class_map[(k,C)] = np.array(np.sort( list(set(temp).intersection(chr_map[CHR]))))
            total += len(class_map[(k,C)])
        print("{0}:{1}, ".format(k+1,total),end='')
    print() # return carrier
    return class_map
  
def var_chunked(X, blocks=10):
    s=[]
    for x in np.array_split(X, blocks, axis=1):
        std_temp = np.var(x, axis=0)
        s.append(std_temp)
    return np.concatenate(s)

def PreparePhenoRHE( Trait, famfile, filename ):
    """
    Create a new tsv file, with labels [FID, IID, Trait] that is aligned with the given fam file,
    as is required for RHEmc. Trait is assumed to be a dataframe, as usually.
    """
    df = pd.read_csv(famfile, sep='\s+', header=None)
    df.set_index(df[0], inplace=True)
    df.drop(columns=[2,3,5,4], inplace=True)
    
    if Trait.ndim==1:
        df[2] = "NA"
        df[2] = Trait
        df.rename( columns={0:"FID", 1:"IID", 2:"Trait"}, inplace=True)
        df.to_csv(filename, index=None, sep='\t', na_rep='NA')
    else:
        df.rename( columns={0:"FID", 1:"IID"}, inplace=True)
        for x in Trait.columns[2:]:
            df[x] = "NA"
            df[x] = Trait[x]
        df.to_csv(filename, index=None, sep='\t', na_rep='NA')
    return 

def regress_out_covariates( covar_file, snp_on_disk, pheno_name ):
    """
    Get the intersection of genotyped and phenotyped samples and regress-out the effects of covariates
    from a given trait. We assume that `pheno_name` and all the covariates are included in the dataframe.
    INPUT : a) path to dataframe, b) Bed format from PySNPTools, c) string
    OUTPUT: a) dataframe (1D) with adjusted trait for the "effective" subset of samples
            b) a list containing the effective subset of samples
    TODO: pass covar_IDs as an argument; load trait from another file?
    """
    df_covar = pd.read_csv(covar_file, sep='\s+', low_memory=False)
    df_covar = df_covar.set_index( df_covar["FID"])
    df_covar = df_covar.dropna(subset=[pheno_name])
    samples_geno = [int(x) for x in snp_on_disk.iid[:,0] ]
    # keep only the intersection of pheno/geno/covar-typed samples
    samples_to_keep = set(samples_geno).intersection(df_covar.FID)
    df_covar = df_covar.drop( set(df_covar['FID']).difference(samples_to_keep), axis=0) 
    samples = {}
    for i in range(len(samples_geno)): 
        if int(snp_on_disk.iid[i,0]) in df_covar.index:
            samples[i] = int(snp_on_disk.iid[i,0])
    # now we need to align the individuals in the df for the math to follow
    df_covar = df_covar.reindex( samples.values() )
    Trait = df_covar[pheno_name]
    N_total = len(Trait)
    # regress out the covariates from the trait
    select = ['Sex', 'Age', 'Site','PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10']
    W = np.concatenate([df_covar[select].to_numpy(), np.ones((N_total,1))], axis=1)
    Trait -= W.dot( np.linalg.inv(W.T.dot(W)) ).dot( W.T.dot(Trait) )
    Trait -= np.mean(Trait); Trait /= np.std(Trait)
    print("Samples with available genotypes, phenotypes, and covariates to keep for analysis", N_total )
    return Trait, list(samples.keys()) 
  
def gen_maf_ld_anno( ldscores_file, model_snps, n_maf_bins=2, n_ld_bins=4, outfile=None ):
    """Intermediate helper function to generate MAF / LD structured annotations. 
    Creates a table of size `len(model_snps)` with values according to the given LDscores file.
    We can either handle number of MAF bins, or bin edges (where n_ld_bins is a list).
    Credits: Arjun (and modified)"""
    print("\nReading the LDscores file and making the annotation...")
    df = pd.read_csv( ldscores_file, sep='\s+', index_col="SNP" )
    # get the intersection of the snps in the bim and the LDscores file
    effective_snps = list()
    map_eff_to_bim = {}
    eff_id = 0
    for i,x in enumerate(model_snps):
        if x in df.index:
            effective_snps.append(x)
            map_eff_to_bim[eff_id] = i
            eff_id += 1
    # the next will be used as a hash-map
    map_eff_to_bim = pd.DataFrame.from_dict(map_eff_to_bim, orient="index")
    
    mafs = df.loc[ effective_snps ].MAF.values
    if "ldscore" in df.columns:
        ld_scores = df.loc[effective_snps].ldscore.values
    elif "LDSCORE" in df.columns:
        ld_scores = df.loc[effective_snps].LDSCORE.values
    else:
        print("ERROR: what's the column of LD-scores?")
    assert(mafs.size == ld_scores.size)
    print("Loaded MAF/LDscores for {0} variants, out of the {1} given. These will be the basis for any model-snps.".format(len(effective_snps), len(df)) )
    del df

    if type(n_maf_bins)==int:
        maf_bins = np.quantile( mafs, np.linspace( 0, 1, n_maf_bins+1 ) )
    else:
        # we assume a list of bounds is passed
        maf_bins = np.copy(n_maf_bins)
        n_maf_bins = len(maf_bins) - 1
        
    ld_score_percentiles = np.linspace( 0, 1, n_ld_bins+1 )
    tot_cats = np.zeros(shape=(len(model_snps), n_maf_bins*n_ld_bins), dtype=np.uint8)

    i = 0
    for j in range(n_maf_bins):
        if j == 0:
            maf_idx = (mafs >= maf_bins[j]) & (mafs < maf_bins[j+1])
        else:
            maf_idx = (mafs >= maf_bins[j]) & (mafs < maf_bins[j+1])
        tru_ld_quantiles = [np.quantile(ld_scores[maf_idx], i) for i in ld_score_percentiles]
        for k in range(n_ld_bins):
            if k == 0:
                ld_idx = (ld_scores >= tru_ld_quantiles[k]) & (ld_scores <= tru_ld_quantiles[k+1])
            else:
                ld_idx = (ld_scores > tru_ld_quantiles[k]) & (ld_scores <= tru_ld_quantiles[k+1])
            cat_idx = np.where(maf_idx & ld_idx)[0]
            # Set the category to one
            tot_cats[ map_eff_to_bim.loc[cat_idx,0].values, i ] = 1
            i += 1
    # Make sure there are SNPs in every category: 
    assert(np.all(np.sum(tot_cats, axis=0) > 0))
    print( "Class sizes:", *np.sum(tot_cats, axis=0) )
    print( "SNPs not in any class:", np.sum( np.sum(tot_cats, axis=1)==0) )
    print( " " )
    if outfile !=None:
        np.savetxt( outfile, tot_cats, fmt='%d')
        print( "Annotation saved in", outfile )
        # np.savetxt( prefix+".maf{0}_ld{1}.annot".format( len(maf_bins)-1, len(ld_score_percentiles)-1 ), tot_cats, fmt='%d')
    return tot_cats

def RunRHE( bedfile, phenoFile, annotation, savelog, rhemc="/data/hornbill/kalantzi/Software/RHE-mc/RHEmc" ):
    """
    A wrapper that prepares a file with SNP annotations (if needed), runs RHE on the background, 
    (always), and returns a list with the estimates. Can deal with any number of components.
    If `annotation` is an integer, we create a new file, otherwise we read from disk.
    We assume K genetic components +1 for the environment. 
    """
    if type(annotation)==str:
        # we assume the annotation exists already and infer K
        df = pd.read_csv(annotation, sep=' ')
        K = df.shape[1]
    if type(annotation)==int:
        # create a new annotation
        _,M = Bed(bedfile, count_A1=True).shape # get the number of SNPs
        K=annotation
        annot = np.random.randint(0,high=K, size=M)
        table = np.zeros((M,K),dtype=np.int8)
        for i in range(M):
            table[i,annot[i]] = 1
        table = pd.DataFrame(table)
        table.to_csv(bedfile+".random.annot",header=None,index=None,sep=' ')
        annotation = bedfile+".random.annot"
            
    # now run RHE
    _ = subprocess.run("rm -f "+savelog+".rhe.log", shell=True )
    cmd = rhemc+" -g "+bedfile+" -p "+phenoFile+" -annot "+annotation+" -k 30 -jn 10 -o "+savelog
    print("Invoke RHE as", cmd)
    _ = subprocess.run(cmd, shell=True)
    
    if os.path.isfile(savelog):
        VC = np.zeros(K+1)
        with open(savelog, 'r') as f:
            lines = f.readlines()
            for k in range(K+1):
                VC[k] = float(lines[k+1].split()[1])
        print("Variance components estimated as",VC)
        return VC
    else:
        print("ERROR: RHEmc did not complete.")
        return "nan"

def fix_rhe_pheno_labels( VC_file, pheno_file ):
    """Fix the output of RHE with the names of phenotypes"""
    VC = pd.read_csv( VC_file, sep='\s+' ) # header=None 
    labels_old = VC.columns
    labels_new = pd.read_csv( pheno_file, sep='\s+').columns[2:]
    
    if VC.iloc[0,0] == labels_new[0]:
        print("The pheno labels look fixed :)")
    else:
        print("Fixing the trait labels in",VC_file)
        for i, x in enumerate(labels_new):
            VC.loc[i,"pheno"] = x
    
        Nbins = (len(VC.columns) -3)//2
        VC.iloc[:,:Nbins+2].to_csv( VC_file, sep='\t', header=None, index=None ) # replace the existing file
    return

def vc_adjustment(VC, thres=0.01):
    """
    adjust the genetic components so that the tiny ones are discarded 
    and the rest can still explain the same total heritability
    """
    ind = VC <= thres # which to discard
    s = np.sum(VC[ind])/(np.sum(~ind)-1) # the per-component correction for the rest
    VCnew = VC + s # increase slightly the good ones
    VCnew[-1] = VC[-1] # keep the same environmental component
    VCnew[ind] = 0 # turn the rest to zero
    return VCnew/np.sum(VCnew)
