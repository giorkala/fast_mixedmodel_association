# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:41:03 2021

@author: kalantzi
"""
from fma_utils import Get_TypeI_rep, Get_TypeII, twoSampZ, get_summary
import numpy as np
import pandas as pd
from scipy.stats import chi2 #, wilcoxon, ttest_ind
import sys
PATH="/FIXTHIS/UKBB_SIM/"

case = sys.argv[1]
print(case)
N_repeats=50

#%% load a random instance of summary statistics
df2 = pd.read_csv(PATH+"design_"+case+"/bolt.0.sumstats.gz", sep='\s+') # just load a SNP index
M = len(df2) #623128 #
ID_noncausal=[]
for C in range(1,6):
    # remember to CHANGE ACCORDING TO SCENARIO! "2*C" or "2*C+1"
    # remember that indexing in sumstats is different than in python
    ID_noncausal.extend(np.where( df2.CHR==2*C )[0])
    
causality = pd.read_csv(PATH+"design_"+case+"/pheno.markers.txt", header=None)
ID_causal = np.where(causality[0]==1)[0]
ID_standa = np.where(causality[0]==2)[0]
print(len(ID_noncausal),len(ID_causal),len(ID_standa))

methods = [  "FMA:C1","FMA:C8", "FMA:C16","BOLT:Inf","BOLT:MoG","Regenie","Regenie+PCA","fastGWA","fastGWA+PCA","Lin Reg", "LinReg+PCA"] 

# arrays to store the important average quantities - just for plotting
mean_chi_causal = np.zeros((N_repeats,len(methods)))
mean_chi_null   = np.zeros((N_repeats,len(methods)))
mean_chi_standa = np.zeros((N_repeats,len(methods)))
lambda_all = np.zeros((N_repeats,len(methods)))
lambda_null= np.zeros((N_repeats,len(methods)))

chisq_all = {}; pvals_all = {}
for met in methods:
    chisq_all[met] = np.zeros((M,N_repeats))
    pvals_all[met] = np.zeros((M,N_repeats))
    
for r in range(N_repeats):
    if "fastGWA" in methods:
        try:
            df2 = pd.read_csv(PATH+"design_"+case+"/fastgwa.{0}.fastGWA.gz".format(r), sep='\s+', compression='gzip')
            chisq_all["fastGWA"][:,r] = chi2.isf( df2.P, 1, 0, 1 )
            pvals_all["fastGWA"][:,r] = df2.P
        except:
            print("WARNING: no results for fastGWA on {0}:{1} were found; assuming null-only instead.".format(case,r))
            chisq_all["fastGWA"][:,r] = np.ones(M)
            pvals_all["fastGWA"][:,r] = np.ones(M)*0.3173
    if "fastGWA+PCA" in methods:
        try:
            df2 = pd.read_csv(PATH+"design_"+case+"/fastgwa_pca.{0}.fastGWA.gz".format(r), sep='\s+', compression='gzip')
            chisq_all["fastGWA+PCA"][:,r] = chi2.isf( df2.P, 1, 0, 1 )
            pvals_all["fastGWA+PCA"][:,r] = df2.P
        except:
            print("WARNING: no results for fastGWA on {0}:{1} were found; assuming null-only instead.".format(case,r))
            chisq_all["fastGWA+PCA"][:,r] = np.ones(M)
            pvals_all["fastGWA+PCA"][:,r] = np.ones(M)*0.3173
    if "BOLT:Inf" in methods:
        df2 = pd.read_csv(PATH+"design_"+case+"/bolt."+str(r)+".sumstats.gz", sep='\s+', compression='gzip')
        chisq_all["BOLT:Inf"][:,r] = df2.CHISQ_BOLT_LMM_INF
        chisq_all["BOLT:MoG"][:,r] = df2.CHISQ_BOLT_LMM
        chisq_all["Lin Reg"][:,r] = df2.CHISQ_LINREG
        pvals_all["BOLT:Inf"][:,r] = df2.P_BOLT_LMM_INF
        pvals_all["BOLT:MoG"][:,r] = df2.P_BOLT_LMM
        pvals_all["Lin Reg"][:,r] = df2.P_LINREG
    if "FMA:C1" in methods:
        df2 = pd.read_csv(PATH+"design_"+case+"/rctamc.C1.pheno{0}.sumstats.gz".format(r), sep='\s+', compression='gzip', usecols=["CHISQ","P"], nrows=M)
        chisq_all["FMA:C1"][:,r] = df2.CHISQ
        pvals_all["FMA:C1"][:,r] = df2.P
    if "FMA:C16" in methods:
        df2 = pd.read_csv(PATH+"design_"+case+"/rctamc.C16.pheno{0}.sumstats.gz".format(r), sep='\s+', compression='gzip', usecols=["CHISQ","P"], nrows=M)
        chisq_all["FMA:C16"][:,r] = df2.CHISQ
        pvals_all["FMA:C16"][:,r] = df2.P
    if "FMA:C8" in methods:
        df2 = pd.read_csv(PATH+"design_"+case+"/rctamc.C8.pheno{0}.sumstats.gz".format(r), sep='\s+', compression='gzip', usecols=["CHISQ","P"], nrows=M)
        chisq_all["FMA:C8"][:,r] = df2.CHISQ
        pvals_all["FMA:C8"][:,r] = df2.P
    if "FMA:C8+PCA" in methods:
        df2 = pd.read_csv(PATH+"design_"+case+"/rctamc_pca.C8.pheno{0}.sumstats.gz".format(r), sep='\s+', compression='gzip', usecols=["CHISQ","P"], nrows=M)
        chisq_all["FMA:C8+PCA"][:,r] = df2.CHISQ
        pvals_all["FMA:C8+PCA"][:,r] = df2.P
    if "LinReg+PCA" in methods:
        df2 = pd.read_csv(PATH+"design_"+case+"/plink.assoc.pheno"+str(r)+".glm.linear.gz", sep='\s+', compression='gzip',)
        df2 = df2[ df2.TEST=="ADD" ]
        chisq_all["LinReg+PCA"][:,r] = df2.T_STAT**2
        pvals_all["LinReg+PCA"][:,r] = df2.P
    if "Regenie" in methods:
        df2 = pd.read_csv(PATH+"design_"+case+"/regenie.all_pheno{0}.regenie.gz".format(r), sep='\s+', usecols=["CHISQ"]) #, compression='gzip'
        chisq_all["Regenie"][:,r] = df2.CHISQ
        pvals_all["Regenie"][:,r] = chi2.sf(df2.CHISQ, df=1)
    if "Regenie+PCA" in methods:
        df2 = pd.read_csv(PATH+"design_"+case+"/regenie_pca.all_pheno{0}.regenie.gz".format(r), sep='\s+', usecols=["CHISQ"]) #, compression='gzip'
        chisq_all["Regenie+PCA"][:,r] = df2.CHISQ
        pvals_all["Regenie+PCA"][:,r] = chi2.sf(df2.CHISQ, df=1)
    # now get the averages
    for i,met in enumerate(methods):
        mean_chi_causal[r,i] = np.mean(chisq_all[met][ID_causal,r])
        mean_chi_null[r,i]   = np.mean(chisq_all[met][ID_noncausal,r])
        lambda_all[r,i] = np.median(chisq_all[met][:,r])/chi2.ppf(0.5, df=1)
        lambda_null[r,i]= np.median(chisq_all[met][ID_noncausal,r])/chi2.ppf(0.5, df=1)
    print(r,end=',')
    
#%% get the statistics
z_tests = {}
Y = np.ones(N_repeats)*0.05 # to compare against H0 for FPR
with open(PATH+"design_"+case+"/comparison_summary.txt", 'w') as f:
    f.write("\n  Method |  Mean@causal |   Mean@null   |       Power      |     TypeI error (se; pval)  \n")
    f.write("-"*103+"\n")
    print  ("\n  Method |  Mean@causal |   Mean@null   |       Power      |     TypeI error   ")
    print("-"*103)
    for met in methods:
        A = chisq_all[met]#[:,35:50]
        FPRS = []; TPRS = []
        for r in range(N_repeats):
            FPRS.append( Get_TypeI_rep( pvals_all[met][:,r], ID_noncausal, [0.05]) )
            TPRS.append( 1 -Get_TypeII( pvals_all[met][:,r], ID_causal, [0.05]))
        z_tests[met] = twoSampZ( np.mean(FPRS), np.mean(Y), 0, np.std(FPRS), np.std(Y), N_repeats, N_repeats )
        if z_tests[met][1] <= 0.05/8: #len(methods)/6:
            f.write(met+" "*(9-len(met))+"| {0:.2f} ({1:.3f}) | {4:.3f} ({5:.3f}) | {6:.3f} ({7:.2e})| {8:.3f}*({9:.2e}; {10:.1e})\n".format( *get_summary(A, ID_causal), *get_summary(A, ID_standa), *get_summary(A, ID_noncausal), np.mean(TPRS), np.std(TPRS)/np.sqrt(N_repeats), np.mean(FPRS), np.std(FPRS)/np.sqrt(N_repeats), z_tests[met][1]) ) # 
            print(met+" "*(9-len(met))+"| {0:.2f} ({1:.3f}) | {4:.3f} ({5:.3f}) | {6:.3f} ({7:.2e})| {8:.3f}* ({9:.2e}; {10:.1e})\n".format( *get_summary(A, ID_causal), *get_summary(A, ID_standa), *get_summary(A, ID_noncausal), np.mean(TPRS), np.std(TPRS)/np.sqrt(N_repeats), np.mean(FPRS), np.std(FPRS)/np.sqrt(N_repeats), z_tests[met][1]) )

        else:
            f.write(met+" "*(9-len(met))+"| {0:.2f} ({1:.3f}) | {4:.3f} ({5:.3f}) | {6:.3f} ({7:.2e})| {8:.3f} ({9:.2e}; {10:.1e})\n".format( *get_summary(A, ID_causal), *get_summary(A, ID_standa), *get_summary(A, ID_noncausal), np.mean(TPRS), np.std(TPRS)/np.sqrt(N_repeats), np.mean(FPRS), np.std(FPRS)/np.sqrt(N_repeats), z_tests[met][1]) ) # 
            print(met+" "*(9-len(met))+"| {0:.2f} ({1:.3f}) | {4:.3f} ({5:.3f}) | {6:.3f} ({7:.2e})| {8:.3f} ({9:.2e}; {10:.1e})\n".format( *get_summary(A, ID_causal), *get_summary(A, ID_standa), *get_summary(A, ID_noncausal), np.mean(TPRS), np.std(TPRS)/np.sqrt(N_repeats), np.mean(FPRS), np.std(FPRS)/np.sqrt(N_repeats), z_tests[met][1]) )

# end-of-script