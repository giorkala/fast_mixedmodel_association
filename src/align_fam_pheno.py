"""
This script creates pheno+covar files that are aligned with a given fam file. Should be used prior to mtRHEmc.
Invoke as `python align_fam_pheno.py geno_prefix phenotypes covariates out`
NOTE: Modify the covariates accordingly.
"""
import pandas as pd
import sys

def align_pheno_with_fam( pheno_file, fam, out=None ):
    ### use this for selecting traits ###
    new = pd.DataFrame()
    new["FID"] = fam[0]; new["IID"] = fam[1]
    new.set_index( new['FID'], inplace=True)

    data = pd.read_csv( pheno_file, sep='\s+' )
    data.set_index(data.FID, inplace=True)
    samples = set(fam[0]).intersection(data.index)

    for x in data.columns[2:]:
        new.loc[samples,x] = data.loc[samples, x]
    
    if out != None:   
        new.to_csv( out, sep='\t', index=None, na_rep="NA" )
    return new

def align_covar_with_fam( covar_file, fam, out=None ):
    ### do this for selecting a few basic covariates ###
    new = pd.DataFrame()
    new["FID"] = fam[0]; new["IID"] = fam[1]
    new.set_index( new['FID'], inplace=True)

    data = pd.read_csv( covar_file, sep='\s+', low_memory=False )
    data.set_index(data.FID, inplace=True)
    samples = set(fam[0]).intersection(data.index)

    select = ["FID","IID"]
    select.extend( ["PCA"+str(i) for i in range(1,11)])
    for x in select:
        new.loc[samples,x] = data.loc[samples, x]
    # new["Age2"] = new.Age**2
    # new["AgeSex"] = new.Age * new.Sex
    # new = new[["FID","IID","Sex","Age","Age2","AgeSex",'PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10', 'PCA11', 'PCA12', 'PCA13', 'PCA14', 'PCA15', 'PCA16', 'PCA17', 'PCA18', 'PCA19', 'PCA20']]
    # new.to_csv("covar_"+sys.argv[4], sep='\t', index=None, na_rep="NA")
    # new = new[["FID","IID","Age","Sex",'PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10']]
    if out != None:
        new.to_csv( out, sep='\t', index=None, na_rep="NA" )
    return new

if __name__ == "__main__":
    fam = pd.read_csv( sys.argv[1]+".fam", sep='\s+', header=None )
    _ = align_pheno_with_fam( sys.argv[2], fam, sys.argv[4]+".pheno" )
    _ = align_covar_with_fam( sys.argv[3], fam, sys.argv[4]+".covar" )
    
# end-of-file