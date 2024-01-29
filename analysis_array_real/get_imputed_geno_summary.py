# -*- coding: utf-8 -*-
"""
Created on Wed May 25 23:55:27 2022

@author: giork
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2

phenotypes = ["BMI","DiastolicBloodPressure","EosinoCount","GlycatedHaemoglobinHbA1c","HDLCholesterol",
              "MeanCorpHaem","MeanCorpVol","MonocyteCount","PlateletCount","RedBloodCellCount",
              "SystolicBloodPressure","TotalCholesterol","WhiteBloodCellCount"]

def snp_clustering_pval( gen_pos, pvalues, r=1e5 ):
    """cluster variants according to genetic position and select the lowest pval.
    The input should contain only the GW-significant variants, for one given chromosome."""
    loci = [ [{gen_pos[0]}, pvalues[0]] ] # a list of (set, best-pval) pairs
    for i in range(1, len(gen_pos)):
        if gen_pos[i] - max(loci[-1][0]) < r:
            loci[-1][0].add( gen_pos[i] )
            loci[-1][1] = min( loci[-1][1], pvalues[i] )
        else:
            loci.append( [ {gen_pos[i]}, pvalues[i] ])
            
    loci_new = []
    for pair in loci:
        if len(pair[0])>1:
            a=min(pair[0])
            b=max(pair[0])
            loci_new.append( (a,b,pair[1]))
        else:
            loci_new.append( [(x+50000,x-50000, pair[1]) for x in pair[0]][0] )
    return loci_new

def get_common_loci( loci_a, loci_b):
    # brute force: shameless iterations
    common = []
    for a in loci_a:
        for b in loci_b:
            if b[0] <= a[0] and a[0] <= b[1]:
                common.append( (a,b) )
            elif a[0] <= b[0] and b[0] <= a[1]:
                common.append( (a,b) )
    return set(common)

df_results = pd.DataFrame(columns=["trait",'method','mean_chi2',"loci","ngwsv","loci_replicated","sv_replicated","sv_repl_ratio","sv_repl_ratio_low"])

for pheno in phenotypes:
    #%% 1. read the files
    df1 = pd.read_csv("/FIXTHIS/."+pheno+".autosome.txt", sep='\s+', index_col=["SNP"])
    
    df_fgwa = pd.read_csv("/FIXTHIS/imputed/imp.fastgwa.{0}.fastGWA.gz".format(pheno), sep='\t', index_col=["SNP"])
    df_fgwa["CHISQ"] = chi2.isf( df_fgwa.P, 1, 0, 1 )
        
    df_rcta = pd.DataFrame(); df_rege = pd.DataFrame(); df_linr = pd.DataFrame(); df_renocv = pd.DataFrame();
    for C in range(1,23):
        df_temp = pd.read_csv("/FIXTHIS/imputed/imp.chr{0}.C8.{1}.sumstats.gz".format(C,pheno), sep='\t') #, index_col=["ID"])
        df_temp.set_index(["ID"], inplace=True, drop=False)
        common0 = set(df_temp.index).intersection(df1.index)
        print(C, len(common0), end=', ')
        df_rcta = df_rcta.append( df_temp.loc[common0] ) #, ignore_index=True)

        df_temp = pd.read_csv("/FIXTHISimputed/imp.chr{0}.regenie_{1}.regenie.gz".format(C,pheno), sep='\s+') #, index_col=["ID"])    
#         df_temp = pd.read_csv("UKBB_REAL/results_ukbb_446k/imputed/imp.chr{0}.regenie_pgen_{1}.regenie.gz".format(C,pheno), sep='\s+') #, index_col=["ID"])    
        df_temp.set_index(["ID"], inplace=True, drop=False)
        common = set(common0).intersection(df_temp.index)
        df_rege = df_rege.append( df_temp.loc[common] )
        
        df_temp = pd.read_csv("/FIXTHIS/imputed/imp.chr{0}.rege_nocov_{1}.regenie.gz".format(C,pheno), sep='\s+') #, index_col=["ID"])    
        df_temp.set_index(["ID"], inplace=True, drop=False)
        common = set(common0).intersection(df_temp.index)
        df_renocv = df_renocv.append( df_temp.loc[common] )
        
    common = set(df1.index).intersection(df_rcta.index)
    for x in [df_rege, df_renocv, df_fgwa]:
        common = common.intersection(x.index)
    
    print('\n', pheno,len(common))
        
    df1 = df1.loc[common]
    df_rcta = df_rcta.loc[common]
    df_rege = df_rege.loc[common]
    df_renocv = df_renocv.loc[common]
    df_fgwa = df_fgwa.loc[common]
    
    #%% 2. get the loci
    total= {"rcta":0, "rege":0, "rege_nocov":0, "fgwa":0, "bbj":0}
    replicated = {"rcta":0, "rege":0, "rege_nocov":0, "fgwa":0}
    for C in range(1,23):
        
        temp = df_rege[df_rege["CHROM"]==C]
        temp = temp.sort_values(by=["GENPOS"])
        temp["P"] = 10**(-temp.LOG10P)
        temp = temp[temp.P <= 5e-8]
        if len(temp)>0:
            loci_rege = snp_clustering_pval( temp.GENPOS, temp.P, 5e5 )
            total["rege"] += len(loci_rege)
            print( "rege:", len(loci_rege), len(temp), end=' | ')
        else:
            print( "rege:", 0, 0, end=' | ')

        temp = df_fgwa[df_fgwa["CHR"]==C]
        temp = temp.sort_values(by=["POS"])
        temp = temp[temp.P <= 5e-8]
        if len(temp)>0:
            loci_fgwa = snp_clustering_pval( temp.POS, temp.P, 5e5 )
            total["fgwa"] += len(loci_fgwa)
            
        temp = df_renocv[df_renocv["CHROM"]==C]
        temp = temp.sort_values(by=["GENPOS"])
        temp["P"] = 10**(-temp.LOG10P)
        temp = temp[temp.P <= 5e-8]
        if len(temp)>0:
            loci_regenocov = snp_clustering_pval( temp.GENPOS, temp.P, 5e5 )
            total["rege_nocov"] += len(loci_regenocov)
        
        temp = df_rcta[df_rcta["#CHROM"]==C]
        temp = temp[temp.P <= 5e-8]
        if len(temp)>0:
            temp = temp.sort_values(by=["POS"])
            loci_rcta = snp_clustering_pval( temp.POS, temp.P, 5e5 )
            total["rcta"] += len(loci_rcta)#
            print( "rcta:", len(loci_rcta), len(temp), end=' | ')
        else:
            print("rcta:", 0, 0, end=' | ')
    
        print("common:", len(get_common_loci( loci_rcta, loci_rege)) )
        
        temp = df1[df1["CHR"]==C]
        temp = temp[temp.P <= 5e-2]
        temp = temp.sort_values(by=["POS"])
        loci_bbj = snp_clustering_pval( temp.POS, temp.P, 2e5 )
        total["bbj"] += len(loci_bbj)
        
        replicated["fgwa"] += len(get_common_loci( loci_fgwa, loci_bbj ))
        replicated["rcta"] += len(get_common_loci( loci_rcta, loci_bbj ))
        replicated["rege"] += len(get_common_loci( loci_rege, loci_bbj ))
        replicated["rege_nocov"] += len(get_common_loci( loci_regenocov, loci_bbj ))
        
    #%% 3. Get the GW summaries
    A = df1[df1.P<=5e-2].index
    
    B = df_fgwa[df_fgwa.P <= 5e-8].index
    Z = df1[df1.P<=5e-2/len(B)].index
    goodA = np.intersect1d(A,B)
    goodZ = np.intersect1d(Z,B)
    df_results = df_results.append( pd.Series({"trait":pheno, 'method':"fastGWA",'mean_chi2':np.mean(df_fgwa.CHISQ),"ngwsv":len(B),
                                               "sv_replicated":len(goodA),"sv_repl_ratio":len(goodA)/len(B),"sv_repl_ratio_low":len(goodZ)/len(B), 
                                               "loci":total["fgwa"], "loci_replicated":replicated["fgwa"]/total["fgwa"]}), ignore_index=True)
     
    B = df_rcta[df_rcta.P <= 5e-8].index
    Z = df1[df1.P<=5e-2/len(B)].index
    goodA = np.intersect1d(A,B)
    goodZ = np.intersect1d(Z,B)
    df_results = df_results.append( pd.Series({"trait":pheno, 'method':"FMA:C8",'mean_chi2':np.mean(df_rcta.CHISQ),"ngwsv":len(B),
                                               "sv_replicated":len(goodA),"sv_repl_ratio":len(goodA)/len(B),"sv_repl_ratio_low":len(goodZ)/len(B), 
                                               "loci":total["rcta"], "loci_replicated":replicated["rcta"]/total["rcta"]}), ignore_index=True)
    
    B = df_rege[df_rege.LOG10P >= -np.log10(5e-8)].index
    Z = df1[df1.P<=5e-2/len(B)].index
    goodA = np.intersect1d(A,B)
    goodZ = np.intersect1d(Z,B)
    df_results = df_results.append( pd.Series({"trait":pheno, 'method':"Regenie",'mean_chi2':np.mean(df_rege.CHISQ),"ngwsv":len(B),
                                               "sv_replicated":len(goodA),"sv_repl_ratio":len(goodA)/len(B),"sv_repl_ratio_low":len(goodZ)/len(B), 
                                               "loci":total["rege"], "loci_replicated":replicated["rege"]/total["rege"]}), ignore_index=True)
    
    B = df_renocv[df_renocv.LOG10P >= -np.log10(5e-8)].index
    Z = df1[df1.P<=5e-2/len(B)].index
    goodA = np.intersect1d(A,B)
    goodZ = np.intersect1d(Z,B)
    df_results = df_results.append( pd.Series({"trait":pheno, 'method':"Rege-nocov",'mean_chi2':np.mean(df_renocv.CHISQ),"ngwsv":len(B),"sv_replicated":len(goodA),
                                               "sv_repl_ratio":len(goodA)/len(B),"sv_repl_ratio_low":len(goodZ)/len(B),
                                               "loci":total["rege_nocov"], "loci_replicated":replicated["rege_nocov"]/total["rege_nocov"]}), ignore_index=True)
    print(df_results.iloc[-3:,:])
    
    #%% 4. finalise
    df_results.to_csv("results.imputed.summary.tab", index=None, sep='\t')
