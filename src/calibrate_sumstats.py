#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script that completes the testing part within the FMA framework. It adjust plink's per chromosome results wrt FMA's calibration, 
either for SNP or imputed data. For the latter it works independently for each chrom, but can be easily modified.

Features
1) Test on SNP data, merge, adjust: `test bed residuals_prefix calibr_prefix output_tag`
2) Merge and adjust stats from imp: `stats prefix input_tag chrom calibrators output_tag`

updates 21/12/21: Assuming that the previous step saves the residuals on a chr-based order, the merging here only works when we have 
sets of residuals obtained from different sets of samples (e.g. IBD-based).
"""
import pandas as pd
import numpy as np
import sys, subprocess
from multiprocessing.pool import Pool as Pool
from scipy.stats import chi2
import time

N_workers = 6
PLINK="/FIXTHIS/plink2"
prefix = sys.argv[2]

if sys.argv[1]=="test":
    print("Get FMA test statistics using PLINK on SNP data.")
    phenotypes = ['BMI', 'MeanPTV', 'ForcedVitalCapacity', 'SystolicBloodPressure', 'DiastolicBloodPressure', 
              'RedBloodCellCount', 'MeanCorpVol', 'MeanCorpHaem', 'RedBloodCellDW', 'WhiteBloodCellCount',
              'PlateletCount', 'PlateletCrit', 'PlateletDW', 'MonocyteCount', 'EosinoCount', 
              'EosinoPercentage', 'MeanSCV', 'TotalCholesterol', 'HDLCholesterol', 'GlycatedHaemoglobinHbA1c']
    # sys.argv: test, bedfile, resid_prefices, calibr_prefices, output_tag
    tic = time.perf_counter()

    # 1. run PLINK (we dont really need to read/change the residuals)
    def multi_run( cmd ):
        subprocess.run( cmd, shell=True )
        return
        
    if sys.argv[2]=="446k":
        sys.argv[2] = "/FIXTHIS/ukb_app43206_500k.bgl5-phased.maf0.01"
        
    POOL = Pool(N_workers) # this controls the chroms which are processed simultaneously so it affects max_RAM a lot
    cmd_all = [PLINK+" --bfile {0} --chr {1} --glm --pheno {2}{1}.gz --threads 2 --memory 8192 --out {3}.chr{1}".format(sys.argv[2],C,sys.argv[3],sys.argv[5]) for C in range(1,23)]
    x = POOL.map( multi_run, cmd_all )
    POOL.close(); POOL.join() 
    
    # 2. prepare the calibrators - they should be prepared already!
    df_cal = pd.read_csv(sys.argv[4]+".calibrators", sep='\t', header=None, index_col=0)
    
    # 3. adjust the test-statistics and save new files
    print("Proceed with adjusting the sumstats")
    
    def adjust_and_save( pheno, header=False ):
        print(pheno, end=', ')
        for C in range(2,23):
            df1 = pd.read_csv("{0}.chr{1}.{2}.glm.linear".format(sys.argv[5],C,pheno), sep='\s+')#, header=None, names=colnames)
            # TODO: maybe we need to check consistency between ref/alt variants? I've noticed opposite effects when DAF>0.50
            # convert T_STAT to CHISQ
            df1["T_STAT"] = df1.T_STAT**2 / df_cal.loc[pheno][1]
            df1.rename(columns={"T_STAT":"CHISQ"}, inplace=True)
            # not sure if it changes much, but we also get new p-values
            df1["P"] = chi2.sf( df1.CHISQ , df=1)
            # convert to better formats for the files
            df1["CHISQ"] = df1["CHISQ"].map( lambda x: '{:.6f}'.format(x))
            df1["P"] = df1["P"].map(lambda x: '{:.2E}'.format(x))
            # 3. save the calibrated sumstats
            df1.to_csv("{0}.{1}.sumstats.gz".format(sys.argv[5],pheno), index=None, sep='\t', compression='gzip', mode='a', header=False)
        return
            
    # we need to process `22 x len(pheno)` files: real old, adjust, save compressed
    # first initialise the new sumstats files using results from chrom-1
    for pheno in phenotypes:
        df1 = pd.read_csv("{0}.chr{1}.{2}.glm.linear".format(sys.argv[5],1,pheno), sep='\s+')#, header=None, names=colnames)
        # convert T_STAT to CHISQ
        df1["T_STAT"] = df1.T_STAT**2 / df_cal.loc[pheno][1]
        df1.rename(columns={"T_STAT":"CHISQ"}, inplace=True)
        # not sure if it changes much, but we also get new p-values
        df1["P"] = chi2.sf( df1.CHISQ , df=1)
        # convert to better formats for the files
        df1["CHISQ"] = df1["CHISQ"].map( lambda x: '{:.6f}'.format(x))
        df1["P"] = df1["P"].map(lambda x: '{:.2E}'.format(x))
        # save the calibrated sumstats directly on disk
        df1.to_csv("{0}.{1}.sumstats.gz".format(sys.argv[5],pheno), index=None, sep='\t', compression='gzip', header=True)#, float_format='%.6f')
        
    print("Processing chroms 2-22...")
    POOL = Pool(N_workers)
    x = POOL.map( adjust_and_save, phenotypes )
    POOL.close(); POOL.join() 
    
    # 4. delete the intermediate files        
    for C in range(1,23):  
        cmd = "rm {0}.chr{1}.*".format(sys.argv[5],C)
        subprocess.run( cmd, shell=True )
    
    print("\nElapsed time for testing 22 chroms = {0:.1f} secs.".format(time.perf_counter()-tic) )  
    
elif sys.argv[1]=="stats":
    try:
        chrom = sys.argv[4]
    except IndexError:
        print("WARNING: No chrom number was passed!")
        chrom = '1'
    print("Adjusting summary statistics for {0} on chrom-{1}".format(prefix,chrom))
    
    tic = time.perf_counter()
    # 1. load/find the calibration factors
    df_cal = pd.read_csv( sys.argv[5], sep='\s+', header=None, index_col=0 )
    
    # we need to process `len(pheno)` files: real old, adjust, save compressed
    for pheno in df_cal.index:
        # 2. load the plink-based sum-stats
#         colnames = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'A1', 'TEST', 'OBS_CT', 'BETA', 'SE', 'T_STAT', 'P']
        df1 = pd.read_csv("{0}/{1}.{2}.glm.linear".format(prefix,sys.argv[3],pheno), sep='\s+')
        # TODO: maybe we need to check consistency between ref/alt variants? I've noticed opposite effects when DAF>0.50
        # convert T_STAT to CHISQ
        df1["T_STAT"] = df1.T_STAT**2 / df_cal.loc[pheno][1]
        df1.rename(columns={"T_STAT":"CHISQ"}, inplace=True)
        # not sure if it changes much, but we also get new p-values
        df1["P"] = chi2.sf( df1.CHISQ , df=1)
        print("GW signif variants = {0} | Mean chisq = {1:.4f} | Lambda gc = {2:.4f} ".format( len(np.where(df1.P<5e-8)[0]), np.mean(df1.CHISQ), np.median( df1.CHISQ )/chi2.ppf(0.5, df=1) ) )
        # convert to better formats for the files
        df1["CHISQ"] = df1["CHISQ"].map( lambda x: '{:.6f}'.format(x))
        df1["P"] = df1["P"].map(lambda x: '{:.2E}'.format(x))
        # 3. save the calibrated sumstats
        df1.to_csv("{0}/{1}.{2}.sumstats.gz".format(prefix,sys.argv[6],pheno), index=None, sep='\t', compression='gzip' )#, float_format='%.6f')
#         df1.to_csv("subsets.{0}.chr{1}.sumstats.gz".format(pheno,chrom), index=None, sep='\t', compression='gzip' )#, float_format='%.6f')
        # TODO: delete the original `*.glm.linear` file
        
    print("Complete! Elapsed time for chrom-{0} = {1:.1f} secs.".format(chrom, time.perf_counter()-tic) )

else:
    print("Unknown option; use `loco` for preparing the LOCO residuals or `stats` for merging the corresponding summary statistics")

# end-of-file
