#!/bin/bash

## OVERVIEW ##
# 0. Use a jupyter notebook to create the lists (UWB, RWB, EUR) of UKBB samples and prepare covariates (`align_fam_pheno.py`).
# 1. Use snakemake (`fma_and_rhe.smk`) to create th bed/bim/fam files, get the ld scores, and create the annotations.
# 2. Run `runthis_simulations.sh sim SAMPLE CASE` to simulate traits
#   SAMPLE is any of UWB, RWB, EUR, CSR; CASE is a075b1 or a075b1_5k
# 3. Apply rhe, rcta, regenie, and bolt on those again using `runthis_simulations.sh METHOD SAMPLE CASE`.
# 4. Analyse the results with `get_summary.py` or another notebook.

date
design=$2
work_dir=/FIXTHIS/UKBB_SIM/design_$design
prefix=/FIXTHIS/UKBB_SIM/DATASETS/ukbb.$3
pheno=$work_dir/phenotypes.tab
echo $prefix

repeats=50
nThreads=5

### WARNINGS ### 
# 1. The genotype/sample QC is not included here
# 2. This script is designed for clusters with an SGE scheduler (ie `qsub`); please modify accordingly 
# 3. Remember to fix the ld-scores file:
#   sed -i '$d' ../PREFIX.score.ld
#   sed -i 's/ldscore/LDSCORE/g' ../$prefix.score.ld (needed for BOLT-LMM)
# 4. Need to exclude the HLA region from the annotation for RHE to work more accurately

### REQUIREMENTS ###
CPU_ARCHITECTURE=$(/FIXTHIS/get-cpu-software-architecture.py) # determine Ivybridge or Skylake compatibility on this node
source /FIXTHIS/mypython-${CPU_ARCHITECTURE}/bin/activate
module load Python/3.7.4-GCCcore-8.3.0
FMA=/FIXTHIS/src
BOLT=/FIXTHIS/BOLT-LMM_v2.3.4/bolt
REGENIE=/FIXTHIS/regenie-3.0.1/regenie
RHE=/FIXTHIS/RHE-mt-new/build/mtRHEmc
PLINK=/FIXTHIS/plink2
GCTA=/FIXTHIS/gcta_1.93.2beta/gcta64
# code for the pheno-simulator https://github.com/alipazokit/simulator; also described at Pazokitoroudi et al. (2020) Nat Comms

if [ $1 = "sim" ]; # 0. Simulate the traits
then
    echo "Simulating traits at $work_dir"
    mkdir $work_dir
    
#     # for x in "UWB" "RWB" "EUR" "XTR"; do sed -i '$d' ukbb.$x\_50k_oddchr.score.ld; done
    echo "p_causal ld_ex maf_ex min_maf max_maf total_h2 num_simul" > $work_dir/param.txt
    echo "0.1 1 0.75 0 0.5 0.25 $repeats" >> $work_dir/param.txt
#     # 5% of 120k is 6000 SNPs or about 1.5% for the 387k
    geno=$prefix\_oddchr # warning: we use the LD-score estimates from the larger cohort
    cut -d' ' -f4,8 $geno.score.ld > $geno.mafld.txt
    cd $work_dir
    /FIXTHIS/Simulator_mafld -g $geno -simul_par param.txt -maf_ld $geno.mafld.txt -annot $geno.uniform.annot -jn 10 -o ./
    if [ -e 0.pheno ]
    then 
        cut -d' ' -f1,2 0.pheno > phenotypes.tab # initialise final file
        for i in {0..49}; do echo "pheno$i" > temp.$i; awk 'NR>1{print $3}' $i.pheno >> temp.$i; done # add proper labels
        paste phenotypes.tab temp.{0..49} > temp # concatenate all phenotypes to the signle file
        mv temp phenotypes.tab
        rm *.pheno temp.*
    fi
    cd ../
fi

if [ $1 = "rhe" ];  # 1. Run RHE on those
then
	cd $work_dir/
	pheno=phenotypes.tab
    if ! [ -e $prefix.covar ]
    then 
        echo "Creating a file with covariates..."
        python $FMA/align_fam_pheno.py $prefix ../Covariates_all.tab
    fi
    # add `-c $prefix.covar` for covariates
 	$RHE -g $prefix -p $pheno -jn 10 -o rhemt.tmp -annot $prefix.uniform.annot -k 50
 	$mysrc/fix_rhemt.sh rhemt.tmp 1 rhe.uniform.all

	$RHE -g $prefix -p $pheno -jn 10 -o rhemt.tmp -annot $prefix.maf2_ld4.annot -k 50
 	$mysrc/fix_rhemt.sh rhemt.tmp 8 rhe.maf2_ld4.all

  	$RHE -g $prefix -p $pheno -jn 10 -o rhemt.tmp -annot $prefix.maf4_ld4.annot -k 50
  	$mysrc/fix_rhemt.sh rhemt.tmp 16 rhe.maf4_ld4.all
    cd ../
fi

if [ $1 = "fma" ];  # 2. Run RCTA on those; assuming that RHE ran succesfully.
then
    covar=$work_dir/dummy_cov.tab # we pass need to pass a dummy covariate (e.g. ones) in case the phenotype is not standardised and mean centered
    $mysrc/invoke_fma_array.py -g $prefix -p $work_dir/phenotypes.tab -c $covar -a $prefix.uniform.annot -o $work_dir/fma.C1 --rheReadLog $work_dir/rhe.uniform.all --nCalibr 2 --SNPtest --maxIters=25 --PoolSize 5 --nThreads $nThreads
    $mysrc/invoke_fma_array.py -g $prefix -p $work_dir/phenotypes.tab -c $covar -a $prefix.maf2_ld4.annot -o $work_dir/fma.C8 --rheReadLog $work_dir/rhe.maf2_ld4.all --nCalibr 2 --SNPtest --maxIters=25 --PoolSize 5 --nThreads $nThreads
    $mysrc/invoke_fma_array.py -g $prefix -p $work_dir/phenotypes.tab -c $covar -a $prefix.maf4_ld4.annot -o $work_dir/fma.C16 --rheReadLog $work_dir/rhe.maf4_ld4.all --nCalibr 2 --SNPtest --maxIters=30 --PoolSize 5 --nThreads $nThreads  --vcThres 0.001
fi

if [ $1 = "bolt" ];  # 3. Run BOLT on those; we can either run it on the raw pheno with covariates, or at the residualised phenotype
then
 	for ((i=0; i<$repeats; i++))
 	do
        echo "Applying Bolt on pheno-$i"
        echo "$BOLT --bfile $prefix --phenoFile=$pheno --phenoCol=pheno$i --lmmForceNonInf --LDscoresFile=$prefix.score.ld --LDscoresCol ldscore --verboseStats --statsFile=$work_dir/bolt.$i.sumstats.gz --numThreads=$nThreads" > bolt.temp
        #  echo "$BOLT --bfile $prefix --phenoFile=$pheno --phenoCol=pheno$i --covarFile=$prefix.covar --qCovarCol=PCA{1:10} --lmmForceNonInf --LDscoresFile=$prefix.ldscores --verboseStats --statsFile=$work_dir/bolt.$i.sumstats.gz --numThreads=$nThreads" > bolt.temp
        qsub -N bolt.$design.$i -pe shmem $nThreads -q short.qc@@short.hge -cwd bolt.temp
 	done
fi

if [ $1 = "regenie" ];  # 4. Run REGENIE on those; we can either run it on the raw pheno with covariates, or without
then
    module load Boost/1.71.0-gompic-2019b
    module load intel/2020a

    tag=regenie
	$REGENIE --step 1 --bed $prefix --phenoFile $pheno --bsize 1000 --extract $prefix.maf001.snps --threads $nThreads --out $work_dir/$tag.all --lowmem --lowmem-prefix $work_dir/tmp.$tag --gz
	$REGENIE --step 2 --bed $prefix --phenoFile $pheno --bsize 500  --threads $nThreads --out $work_dir/$tag.all --minMAC 1 --pred $work_dir/$tag.all\_pred.list --gz

    tag=regenie_pca
    covar=$prefix.covar
	$REGENIE --step 1 --bed $prefix --phenoFile $pheno --covarFile $covar --bsize 1000 --threads $nThreads --out $work_dir/$tag.all --lowmem --lowmem-prefix $work_dir/regenie_tmp --gz
	$REGENIE --step 2 --bed $prefix --phenoFile $pheno --covarFile $covar --bsize 500 --threads $nThreads --out $work_dir/$tag.all --minMAC 1 --pred $work_dir/$tag.all\_pred.list --gz
fi

if [ $1 = "linreg" ]; # 5. Run LINREG+PCA as implemented in PLINK (LINREG without PCs can be obtained from BOLT-LMM)
then
	$PLINK --bfile $prefix --pheno $work_dir/phenotypes.tab --covar $prefix.covar --covar-number 3-7 --covar-variance-standardize --glm hide-covar --out $work_dir/plink.assoc --threads $nThreads
    echo "Compressing sum-stats files..."
    for x in $work_dir/plink.assoc.pheno*.glm.linear; do gzip $x; done
    echo "all done!"
fi

if [ $1 = "fastgwa" ]; # 6. Run FastGWA, again with or without conditioning to PCs
then
    if ! [ -e $prefix.grm.sp ]
    then
        echo "Creating a GRM with GCTA"
        $GCTA --bfile $prefix --make-grm-part 3 1 --thread-num 3 --out $prefix
        $GCTA --bfile $prefix --make-grm-part 3 2 --thread-num 3 --out $prefix
        $GCTA --bfile $prefix --make-grm-part 3 3 --thread-num 3 --out $prefix
        cat $prefix.part_3_*.grm.id > $prefix.grm.id
        cat $prefix.part_3_*.grm.bin > $prefix.grm.bin
        cat $prefix.part_3_*.grm.N.bin > $prefix.grm.N.bin
        $GCTA --grm $prefix --make-bK-sparse 0.05 --out $prefix --thread-num 3
        rm $prefix*part*
    fi
    
    cd $work_dir
    rm -f fastgwa.gamma
	for ((i=0; i<$repeats; i++))
	do
        echo "Applying fastGWA on pheno-$i"
        awk -v c=$(($i+3)) '{print $1, $2, $c}' phenotypes.tab > pheno.$i.gcta
        echo "$GCTA --bfile $prefix --grm-sparse $prefix --fastGWA-mlm --pheno pheno.$i.gcta --threads 5 --out fastgwa.$i" > fastgwa.temp
        echo "gzip -f fastgwa.$i.fastGWA" >> fastgwa.temp 
        echo "cat fastgwa.$i.fastGWA.gamma >> fastgwa.gamma" >> fastgwa.temp 
        qsub -N fastgwa.$design.$i -pe shmem $nThreads -q short.qc@@short.hge -cwd fastgwa.temp

        echo "Applying fastGWA+PCA on pheno-$i"
        # awk -v c=$(($i+3)) '{print $1, $2, $c}' phenotypes.tab > pheno.$i.gcta
        echo "$GCTA --bfile $prefix --grm-sparse $prefix --fastGWA-mlm --pheno pheno.$i.gcta --qcovar ../Covariates_5pcs.gcta --threads 5 --out fastgwa_pca.$i" > fastgwa.temp
        echo "gzip -f fastgwa_pca.$i.fastGWA" >> fastgwa.temp
        qsub -N fastgwa_pca.$design.$i -pe shmem $nThreads -q short.qc@@short.hge -cwd fastgwa.temp
    done
    cd ../
fi

date

# submit as follows
# method="rcta"; for a in UWB RWB EUR CSR; do for b in "a075b1" "a075b1_4k"; do qsub -N $method.$a\_$b -pe shmem 5 -q short.qc@@short.hge -cwd runthis_simulations.sh $method $a\_$b $a\_50k; done; done
