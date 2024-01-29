# -*- coding: utf-8 -*-
"""
Created on Fri May 29 18:44:40 2020

@author: giork
"""

import pandas as pd
import numpy as np

# Generate a MAF/LD annotation file
def gen_maf_ld_anno(ld_score_file, maf_bins, ld_score_percentiles, outfile=None):
    """Intermediate helper function to generate MAF / LD structured annotations. 
    This is slightly different from the one in `rcta/main.py`
    Credits: Arjun"""
    df = pd.read_csv(ld_score_file, sep='\s+')
    mafs = df.MAF.values
    ld_scores = df.ldscore.values
    assert(mafs.size == ld_scores.size)
    # subsetting to variants in the correct MAF bin
    n_maf_bins = len(maf_bins) - 1
    n_ld_bins = len(ld_score_percentiles) - 1
    tot_cats = np.zeros(shape=(mafs.size, n_maf_bins*n_ld_bins), dtype=np.uint8)
    i = 0
    for j in range(n_maf_bins):
        if j == 0:
            maf_idx = (mafs > maf_bins[j]) & (mafs <= maf_bins[j+1])
        else:
            maf_idx = (mafs > maf_bins[j]) & (mafs <= maf_bins[j+1])
        tru_ld_quantiles = [np.quantile(ld_scores[maf_idx], i) for i in ld_score_percentiles]
        for k in range(n_ld_bins):
            if k == 0:
              ld_idx = (ld_scores >= tru_ld_quantiles[k]) & (ld_scores <= tru_ld_quantiles[k+1])
            else:
              ld_idx = (ld_scores > tru_ld_quantiles[k]) & (ld_scores <= tru_ld_quantiles[k+1])
            cat_idx = np.where(maf_idx & ld_idx)[0]
            # Set the category to one
            tot_cats[cat_idx,i] = 1
            i += 1
    #   Make sure there are SNPs in every category 
    assert(np.all(np.sum(tot_cats, axis=0) > 0))
    np.savetxt(outfile, tot_cats, fmt='%d')
    
def gen_uniform_anno(ld_score_file, outfile):
    """Generate a uniform annotation for RHE with a single-component. Credits: Arjun"""
    ld_df = pd.read_csv(ld_score_file, sep=' ')
    maf = ld_df.MAF.values
    anno = np.zeros(maf.size, dtype=np.uint8)
    idx = np.where(maf > 0)[0]
    anno[idx] = 1
    np.savetxt(outfile, anno.T, fmt='%d')
    
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

#%% tools for stat evaluation of experiments with synthetic traits
def Get_TypeI_rep( pvalues, noncausal_vars, alpha):
    # False positives; FPR
    if pvalues.ndim > 1:
        FP = 0
        for i in range(pvalues.shape[1]):
            FP += Get_TypeI( pvalues[:,i], noncausal_vars, alpha)
        return FP/pvalues.shape[1]
    else:
        return Get_TypeI( pvalues, noncausal_vars, alpha)
    
def Get_TypeI( pvalues, noncausal_vars, alpha):
    # False positives; FPR
    if len(alpha)==1:
        FP = len(np.where(pvalues[noncausal_vars] < alpha[0])[0])
        return FP/len(noncausal_vars)
    else:
        results = []
        for a in alpha:
            results.append( len(np.where(pvalues[noncausal_vars] < a)[0])/len(noncausal_vars) )
        return np.array(results)    
        
def Get_TypeII_rep( pvalues, causal_vars, signif):
    # False negative; FNR; 1-power
    if pvalues.ndim > 1:
        FN = 0
        for i in range(pvalues.shape[1]):
            FN += Get_TypeII( pvalues[:,i], causal_vars, signif)
        return FN/pvalues.shape[1]
    else:
        return Get_TypeII( pvalues, causal_vars, signif)

def Get_TypeII( pvalues, causal_vars, signif):
    # False negative; FNR; 1-power
    if len(signif)==1:
        FN = len(np.where(pvalues[causal_vars] > signif[0])[0])
        return FN/len(causal_vars)
    else:
        results = []
        for a in signif:
            results.append( len(np.where(pvalues[causal_vars] > a)[0])/len(causal_vars) )
        return np.array(results)
    
def twoSampZ(X1, X2, mudiff, sd1, sd2, n1, n2):
    "credits https://stats.stackexchange.com/questions/124096/two-samples-z-test-in-python"
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    pooledSE = sqrt(sd1**2/n1 + sd2**2/n2)
    z = ((X1 - X2) - mudiff)/pooledSE
    pval = 2*(1 - norm.cdf(abs(z)))
    return round(z, 3), round(pval, 4)

def bootstrap_mean(data, quant=0.95, replicates=10000):
    """Credits: Pier"""
    np.random.seed(1234)
    import scipy.stats.mstats as ms
    stats = np.zeros((replicates,1))
    for i in range(replicates):
        boots = np.random.choice(data, len(data))
        stats[i] = np.average(boots)
    CI = ms.hdquantiles(stats, prob=[0.025, 0.5, 0.975])
    return np.average(data), CI[0], CI[1], CI[2], len(data)

def bootstrap_compare_mean(data1, data2, replicates=10000):
    """Credits: Pier"""
    lt=0
    for i in range(replicates):
        boots1 = np.random.choice(data1, len(data1))
        boots2 = np.random.choice(data2, len(data2))
        if np.average(boots1) < np.average(boots2):
            lt = lt+1
    return lt/replicates, replicates

def pairedTtest( X1, X2, upper=True ):#, sd1, sd2):
    from scipy.stats import t
    n = X1.shape[0]
    d = (np.array(A) - np.array(B)).astype(np.float64)
    pooledSE = np.sqrt( np.var(d, ddof=1)/n ) # same as sqrt( sd^2 / n )  
    score = ( np.mean(d) )/pooledSE
    if upper:
        pval = 1 - t.cdf(np.abs(score), df=n-1)
    else:
        pval = 2*(1 - t.cdf(np.abs(score), df=n-1))
    return score, pval

def get_summary( *args ):
    if len(args)==1:
        A = np.nanmean( args[0], axis=0 )
    else:
        A = np.nanmean( args[0][args[1],:], axis=0 )            
    return np.nanmean(A), np.nanstd(A)/np.sqrt(args[0].shape[1])


#%%
# Generate a MAF/LD annotation file
def gen_maf_ld_anno(ld_score_file, maf_bins, ld_score_percentiles, outfile=None):
  """Intermediate helper function to generate MAF / LD structured annotations. Credits: Arjun"""
  df = pd.read_csv(ld_score_file, sep='\s+')
  mafs = df.MAF.values
  ld_scores = df.ldscore.values
  assert(mafs.size == ld_scores.size)
  # subsetting to variants in the correct MAF bin
  n_maf_bins = len(maf_bins) - 1
  n_ld_bins = len(ld_score_percentiles) - 1
  tot_cats = np.zeros(shape=(mafs.size, n_maf_bins*n_ld_bins), dtype=np.uint8)
  i = 0
  for j in range(n_maf_bins):
    if j == 0:
      maf_idx = (mafs > maf_bins[j]) & (mafs <= maf_bins[j+1])
    else:
      maf_idx = (mafs > maf_bins[j]) & (mafs <= maf_bins[j+1])
    tru_ld_quantiles = [np.quantile(ld_scores[maf_idx], i) for i in ld_score_percentiles]
    for k in range(n_ld_bins):
      if k == 0:
        ld_idx = (ld_scores >= tru_ld_quantiles[k]) & (ld_scores <= tru_ld_quantiles[k+1])
      else:
        ld_idx = (ld_scores > tru_ld_quantiles[k]) & (ld_scores <= tru_ld_quantiles[k+1])
      cat_idx = np.where(maf_idx & ld_idx)[0]
      # Set the category to one
      tot_cats[cat_idx,i] = 1
      i += 1
  # Do some sanity checks before passing along to RHE
  #   Make sure there are SNPs in every category 
  assert(np.all(np.sum(tot_cats, axis=0) > 0))
  np.savetxt(outfile, tot_cats, fmt='%d')
    
def gen_uniform_anno(ld_score_file, outfile):
  """Generate a uniform annotation for RHE with a single-component. Credits: Arjun"""
  ld_df = pd.read_csv(ld_score_file, sep=' ')
  maf = ld_df.MAF.values
  anno = np.zeros(maf.size, dtype=np.uint8)
  idx = np.where(maf > 0)[0]
  anno[idx] = 1
  np.savetxt(outfile, anno.T, fmt='%d')
