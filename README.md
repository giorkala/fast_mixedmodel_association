# FMA: Fast Mixed-model Association
An efficient implementation of linear mixed models for genome-wide association studies (GWAS), supporting multiple genetic components. `FMA` is introduced in Chapter 4 of my [thesis](https://ora.ox.ac.uk/objects/uuid:3d568910-241a-4350-9df8-93a847c0a696), where I present benchmarks for FMA and other state-of-the-art GWAS methods. `FMA` is scalable, as we don't load genotypes, and fast, as we can analyse multiple traits simultaneously. 
This repository contains all the code needed there, including scripts for simulations and analyses with real phenotypes. In brief, the code is partitioned in 3 directories:
1. `src`: contains all the basic modules to run FMA for array data;
2. `analysis_array_sims`: scripts (bash and python) to run FMA and other methods for simulations based on UKBB SNP data;  
3. `analysis_array_real`: scripts (bash and python) to run FMA and other methods on UKBB SNP real data;

#### How to run with Snakemake (recommended)
`snakemake -j1 -s fma_and_rhe.smk run_all [-n]`  
This starts from a bed/bim/fam dataset and a file with phenotypes. It creates an annotation (based on a given set of options), runs RHE to get h2 estimates, and then runs FMA for association. It can also test imputed genotypes, assuming the availability of PGEN files. Check the .smk file for more details about input and how to modify paths and select tasks.

#### How to run when everything else is ready
```
python src/run_fma.py \
	-g $geno \ # the bed file
	-p $pheno \ # the phenotypes file (header: FID IID pheno1 pheno2 ...)
	-c $covar \ # covariates (header: FID IID pheno1 pheno2 ...) (optional)
	-a $geno.maf2_ld4.annot \ # file with annotation; one column per component; no overlapping
	-o fma.C8 \ # prefix for any output
	--rhe \ # run RHE-mc to get VC estimates; alternatively:
	--rheReadLog VC_estimates.txt \ file with VC estimates (phenotypes x classes)
	--SNPtest \ # test for association on the current set of SNPs (optional)
	--nCalibr 3 \ # number of variants per chromosome to estimate the calibration
	--maxIters 50 \ # max number of conjugate gradient iterations (depends on N and sample structure)
	--PoolSize 5 \ # number of chunks to process in parallel
	--nThreads 8  # number of CPU cores to use
```

### Notes for running and reproducing the results
* Several scripts require paths to software or data which can be spotted with the prefix "FIXTHIS".
* To install the multiple-trait version of RHEmc, refer to this [repo](https://github.com/alipazokit/multi-trait/tree/main).
* Simulations with UK Biobank array data (following my thesis): 
```
bash analysis_array_sims/runthis_simulations.sh {method} {case} {sample}
```
> `method` should be any among sim/rhe/fma/bolt/regenie/fastgwa/linreg, `case` is "a075b1_4k" or "a075b1" (architecture), and `sample` is any from UWB/RWR/EUR/CSR.
* Simulations to assess h2 estimation: `bash analysis_array_sims/runthis_h2.sh N` (one shot for all methods, for a given sample size `N`).
* GWAS using real UKBB phenotypes: run `bash analysis_array_real/runthis_real20.sh fma 446k`, or similarly for other methods and samples. Then analyse the results with `get_imputed_geno_summary.py` or similar tools.
* Note many parts of these are designed for using within an SGE cluster (ie `qsub`). 
* You can calculate within-cohort LD scores with GCTA as 
```
$gcta --bfile {geno_prefix} --thread-num 3 --ld-score --ld-wind 200 --out {geno_prefix}"
sed -i '$d' {prefix}.score.ld") # remove the last line as its empty and might cause other programs to crash
```

### Workflow for core calculations for SNP model-fitting
`run_fma.py` -> `FMA_streaming_mt` -> `solve_conj_grad` -> `covar_matmat` -> `core_calc_mp`
The last part is the hardest module as it performs most calculations, and the first one is a wrapper that connects all parts of the framework. That's for the `--lowMem` option; a similar workflow holds for the full mem approach but is not finalised (ie not very modular).

### Tasks needed for simulations with UKBB SNP data
1. [GCTA] Get LD scores and MAF per SNP (once, optional)
2. [Python] Combine the above and create annotations (many options available)
3. [RHEmc] Use the bedfile +phenotype +annotation and get VC estimates (annot+trait specific)
4. [Python] Create the hdf5 (once for each working sample)
5. [Python] Calculate residuals and estimate the calibration factor (main FMA module)
6. [Python or PLINK] Calculate test statistics for SNP, imputed data, dosages etc
