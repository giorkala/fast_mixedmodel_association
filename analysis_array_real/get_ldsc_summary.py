"""
Custom script to perform weighted LD-score regression as in BOLT-LMM
Note: we assume and use base pair location instead of genetic position 

credits: Hrushi
modified by: Yiorgos (May 11th, 2022)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import copy
from multiprocessing.pool import Pool as Pool

outlier_window = 1e6
minMAF = 0.01
outlierVarFracThresh = 0.001
min_outlier_chisq_thresh = 20.0

bfile = "/FIXTHIS/ukb_app43206_500k.bgl5-phased.maf0.01"
ldscores_file = "/FIXTHIS/ldsc/LDSCORE.1000G_EUR.tab.gz"

phenotypes = ['BMI', 'MeanPTV', 'ForcedVitalCapacity', 'SystolicBloodPressure', 'DiastolicBloodPressure', 
              'RedBloodCellCount', 'MeanCorpVol', 'MeanCorpHaem', 'RedBloodCellDW', 'WhiteBloodCellCount',
              'PlateletCount', 'PlateletCrit', 'PlateletDW', 'MonocyteCount', 'EosinoCount', 
              'EosinoPercentage', 'MeanSCV', 'TotalCholesterol', 'HDLCholesterol', 'GlycatedHaemoglobinHbA1c']

def get_mask_dodgy(ldscores, sumstats, bfile):
    """
    Returns mask on sumstats file based on which SNPs to keep
    """
    N = 446050 # Bed(bfile, count_A1=True).shape[0]
    num_snps = len(sumstats)
    mask_dodgy = np.ones(num_snps, dtype=bool)
    pos_column = "POS"

    sumstats["CHISQ_FLT"] = sumstats["CHISQ"].astype("float")
    sumstats = sumstats.sort_values("POS")
    chisq_thresh = max(min_outlier_chisq_thresh, N * outlierVarFracThresh)
#    print("Removing SNPs above chisq " + str(chisq_thresh))
    snp_above_chisqth = np.where(sumstats["CHISQ_FLT"] >= chisq_thresh)[0]

    pos_arr = sumstats[pos_column].values
    maf_arr = sumstats["ALT_FREQS"].values
    chisq_arr = sumstats["CHISQ_FLT"].values
    if len(snp_above_chisqth) > 0:
        count = snp_above_chisqth[0]

    for snp in range(num_snps):
        if maf_arr[snp] < minMAF or maf_arr[snp] > 1 - minMAF:
            mask_dodgy[snp] = False
        if np.isnan(chisq_arr[snp]):
            mask_dodgy[snp] = False
        if len(snp_above_chisqth) > 0:
            if np.abs(pos_arr[snp] - pos_arr[count]) < outlier_window:
                mask_dodgy[snp] = False
            else:
                if snp > count:
                    count += 1

    df_all = sumstats.merge(
        ldscores.drop_duplicates(), on="SNP", how="left", indicator=True
    )

    mask_dodgy2 = (df_all["_merge"] == "both").values
#    print( "Number of SNPs remaining after filtering = " + str(np.sum(mask_dodgy * mask_dodgy2)) )
    return mask_dodgy * mask_dodgy2

def ldscore_intercept( ldscores, sumstats, mask_dodgy, tag ):
    """
    Masking SNPs based on:
    1. availability in LDscore file and sumstats file
    2. MAF in our dataset >= 0.01
    3. Remove nearby and top 0.1% (or SNPs with CHISQ >= 20) based on CHISQ
    """
    sumstats = copy.deepcopy(sumstats)
    if "SNP" not in sumstats.columns:
        sumstats.rename(columns={"ID": "SNP"}, inplace=True)
    sumstats = sumstats.sort_values("POS")
    sumstats["CHISQ_FLT"] = sumstats["CHISQ"].astype("float")

    chisq_arr = sumstats["CHISQ_FLT"].values[mask_dodgy]
    ldscore_arr = pd.merge(sumstats, ldscores.drop_duplicates(), on="SNP", how="left")[
        "LDSCORE"
    ].values[mask_dodgy]
#    ldscore_chip_arr = np.array(
#        ldscore_chip["LDSCORE_CHIP"].values[mask_dodgy], dtype="float"
#    )
    if np.sum(mask_dodgy) < len(mask_dodgy) // 2:
        raise Warning("More than half SNPs removed by high CHISQ thresholding")
    print(tag, "Number of SNPs available with LDscores = " + str(len(ldscore_arr)))

    """
    Perform a weighted linear regression of CHISQ with LDSCORES to get intercept and slope
    """
    new_mask = ~np.isnan(ldscore_arr)
    slopeToCM = (np.mean(chisq_arr[new_mask]) - 1) / np.mean(ldscore_arr[new_mask])
    weight = (1 / np.maximum(1, 1 + slopeToCM * ldscore_arr[new_mask])) ** 2
    weight *= 1 / np.maximum(1, ldscore_arr[new_mask])
    wls_model = sm.WLS(
        chisq_arr[new_mask], sm.add_constant(ldscore_arr[new_mask]), weights=weight
    )
    results = wls_model.fit()
    return results.params[0], np.mean(chisq_arr[new_mask])
  
def get_estimates_mp( pheno ):
    df_temp = pd.DataFrame(columns=["pheno","method","gwsv","mean_chi2","mean_chi2_masked","intercept","ratio"])
    
    sumstats = pd.read_csv("/FIXTHIS/results_446k/rctamc.C8."+pheno+".sumstats.gz", sep='\s+')
    sumstats = sumstats.rename(columns={"BP": "POS", "AFREQ": "ALT_FREQS"})
    mask_dodgy = get_mask_dodgy(ldscores[["SNP", "LDSCORE"]], sumstats, bfile)
    intercept, mean_sumstats = ldscore_intercept( ldscores, sumstats, mask_dodgy, pheno+" @ "+"RCTA" )
    atten_ratio = (intercept - 1) / (mean_sumstats - 1)
    df_temp = df_temp.append( pd.Series({"pheno":pheno, 
                                  'method':"RCTA:C8",
                                  'mean_chi2':sumstats.CHISQ.mean(), 
                                  'mean_chi2_masked':mean_sumstats,
                                  "gwsv":len(np.where( sumstats.CHISQ > 29 )[0]), 
                                  "intercept":intercept, 
                                  "ratio":atten_ratio }), 
                      ignore_index=True)

    sumstats = pd.read_csv("/FIXTHIS/results_446k/regenie.all_"+pheno+".regenie.gz", sep='\s+')
    sumstats = sumstats.rename(columns={"A1FREQ": "ALT_FREQS", "GENPOS":"POS", "ID":"SNP" })
    mask_dodgy = get_mask_dodgy(ldscores[["SNP", "LDSCORE"]], sumstats, bfile)
    intercept, mean_sumstats = ldscore_intercept( ldscores, sumstats, mask_dodgy, pheno+" @ "+"Rege" )
    atten_ratio = (intercept - 1) / (mean_sumstats - 1)
    df_temp = df_temp.append( pd.Series({"pheno":pheno, 
                                  'method':"Regenie",
                                  'mean_chi2':sumstats.CHISQ.mean(), 
                                  'mean_chi2_masked':mean_sumstats,
                                  "gwsv":len(np.where( sumstats.CHISQ > 29 )[0]), 
                                  "intercept":intercept, 
                                  "ratio":atten_ratio }), 
                      ignore_index=True)

    sumstats = pd.read_csv("/FIXTHIS/results_446k/bolt."+pheno+".sumstats.gz", sep='\s+')
    sumstats = sumstats.rename(columns={"BP": "POS","A1FREQ": "ALT_FREQS", "CHISQ_BOLT_LMM":"CHISQ" })
    mask_dodgy = get_mask_dodgy(ldscores[["SNP", "LDSCORE"]], sumstats, bfile)
    intercept, mean_sumstats = ldscore_intercept( ldscores, sumstats, mask_dodgy, pheno+" @ "+"BOLT" )
    atten_ratio = (intercept - 1) / (mean_sumstats - 1)
    df_temp = df_temp.append( pd.Series({"pheno":pheno, 
                                  'method':"BOLT:MoG",
                                  'mean_chi2':sumstats.CHISQ.mean(), 
                                  'mean_chi2_masked':mean_sumstats,
                                  "gwsv":len(np.where( sumstats.CHISQ > 29 )[0]), 
                                  "intercept":intercept, 
                                  "ratio":atten_ratio }), 
                      ignore_index=True)
    
    #finally, get LINREG stats from BOLT's file
    sumstats = pd.read_csv("/FIXTHIS/results_328k/bolt."+pheno+".sumstats.gz", sep='\s+')
    sumstats = sumstats.rename(columns={"BP": "POS","A1FREQ": "ALT_FREQS","CHISQ_LINREG":"CHISQ"})
    mask_dodgy = get_mask_dodgy(ldscores[["SNP", "LDSCORE"]], sumstats, bfile)
    intercept, mean_sumstats = ldscore_intercept( ldscores, sumstats, mask_dodgy, pheno+" @ "+"BOLT" )
    atten_ratio = (intercept - 1) / (mean_sumstats - 1)
    df_temp = df_temp.append( pd.Series({"pheno":pheno, 
                                  'method':"LR@328k",
                                  'mean_chi2':sumstats.CHISQ.mean(), 
                                  'mean_chi2_masked':mean_sumstats,
                                  "gwsv":len(np.where( sumstats.CHISQ > 29 )[0]), 
                                  "intercept":intercept, 
                                  "ratio":atten_ratio }), 
                      ignore_index=True)
    return df_temp

if __name__ == "__main__":
    ldscores = pd.read_csv( ldscores_file, sep="\s+" )
    
    POOL = Pool(4)
    results = POOL.map( get_estimates_mp, (p for p in phenotypes))
    POOL.close (); POOL.join () # close; do we need this??  
    
    df_summary = pd.DataFrame(columns=["pheno","method","gwsv","mean_chi2","mean_chi2_masked","intercept","ratio"])
    for x in results:
        df_summary = df_summary.append(x, ignore_index=True)
    df_summary.to_csv("summary_ldsc.tab", sep='\t', index=None)
    
# end-of-file