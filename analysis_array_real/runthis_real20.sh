#!/bin/bash

# ANALYSIS OF UKBB PHENOTYPES #
# we run each method on 20 popular UKBB phenotypes, on increasing sets of individuals

## requirements
source /well/palamara/users/jua009/mypython/mypython-${CPU_ARCHITECTURE}/bin/activate
module load Python/3.7.4-GCCcore-8.3.0
FMA=/FIXTHIS/src
BOLT=/FIXTHIS/BOLT-LMM_v2.3.4/bolt
REGENIE=/FIXTHIS/regenie-3.0.1/regenie
RHE=/FIXTHIS/RHE-mt-new/build/mtRHEmc
PLINK=/FIXTHIS/plink2
GCTA=/FIXTHIS/gcta_1.93.2beta/gcta64

# input files 
work_dir=results_$2
geno=/FIXTHIS/ukb.phased.maf0.01
PGEN=/FIXTHIS/imputed_genotypes_V3/ukb_imp
annoPrefix=/FIXTHIS/ukb.phased.maf0.01.annot
ldPrefix=/FIXTHIS/ukb.phased.maf0.01.score.ld # this is needed for BOLT; we also have in-cohort estimates
pheno=$work_dir/phenotypes.tab
covar=$work_dir/covariates.tab


if [ $1 = "rhe" ]; # 1. Run RHE on those
then
	$RHE -g $geno -p $pheno -c $covar -jn 10 -o rhemt.tmp -annot $annoPrefix.maf2_ld4.annot -k 50
	$RHE -g $geno -p $pheno -c $covar -jn 10 -o rhemt.tmp -annot $annoPrefix.maf4_ld4.annot -k 50
	$RHE -g $geno -p $pheno -c $covar -jn 10 -o rhemt.tmp -annot $annoPrefix.uniform.annot -k 50
fi

[ $1 = "fma" ]; # 2. Run FMA on those
then
	# for the MT version, we first need to run RHE using all the traits and then create the rhe*all file with all the VCs.
    # WARNING: make sure that the trait labels in rhe.*.all are correct. ALSO: "446" should use the original bed/bim/fam files
    # recommended CG limits: 20,40,50,60
 	$mysrc/invoke_mt.py -g $geno -p $pheno -c $covar -a $annoPrefix.maf2_ld4.annot --rheReadLog $work_dir/rhe.maf2_ld4.all -o $work_dir/rctamc.C8  --SNPtest --PoolSize 6 --nCalibr 1 --maxIters=20 --nThreads 12 #--debug #--useHDF
#  	$mysrc/invoke_mt.py -g $geno -p $pheno -c $covar -a $annoPrefix.maf4_ld4.annot --rheReadLog $work_dir/rhe.maf4_ld4.all -o $work_dir/rctamc.C16 --SNPtest --PoolSize 6 --nCalibr 1 --maxIters=40 --nThreads 12 # --debug
# 	$mysrc/invoke_mt.py -g $geno -p $pheno -c $covar -a $annoPrefix.uniform.annot  --rheReadLog $work_dir/rhe.uniform.all  -o $work_dir/rctamc.C1  --SNPtest --PoolSize 6 --nCalibr 1 --maxIters=53 --nThreads 12

#  	$mysrc/invoke_mt.py -g $geno -p $pheno -c $covar -a $annoPrefix.uniform.annot --rheReadLog $work_dir/rhe.uniform.all --faster -o $work_dir/rctamc.Fast --SNPtest --PoolSize 6 --nCalibr 0 --maxIters=50 --nThreads 12
#  	$mysrc/invoke_mt.py -g $geno -p $pheno -c $covar -a $work_dir/rctamc.pruned.annot --rheReadLog $work_dir/rctamc.pruned.rhe -o $work_dir/rctamc.pruned --SNPtest --PoolSize 6 --nCalibr 1 --maxIters=45 --nThreads 12 # --debug
    
#     $mysrc/invoke_mt.py -g $geno -p $work_dir/pheno_ibdsubset$3.tab -c $covar -a $annoPrefix.maf4_ld4.annot --vcThres 0.01 --rheReadLog $work_dir/rhe.maf4_ld4.all -o $work_dir/rctamc.C16_fastibds$3 --PoolSize 8 --nCalibr 1 --maxIters=30 --nThreads 12 --useHDF $work_dir/rctamc.C16_fastibds$3.hdf5
#     $mysrc/invoke_mt.py -g $geno -p $work_dir/pheno_subset_$3_of_6.tab -c $covar -a $annoPrefix.maf4_ld4.annot --rheReadLog results_ukbb_50k/rhe.maf4_ld4.all -o $work_dir/rctamc.C16_ss6_$3.b --useHDF $work_dir/rctamc.C16_ss6_$3.hdf5 --PoolSize 10 --nCalibr 1 --maxIters=25 --nThreads 12
#    mprof run --include-children $mysrc/invoke_mt.py -g $geno -p $pheno -a $annoPrefix.maf4_ld4.annot -o $work_dir/rctamc.C16 --rheReadLog $work_dir/rhe.maf4_ld4.all --PoolSize 1 --nCalibr 1 --maxIters=10  --nThreads 12 --debug
fi

[ $1 = "regenie" ];# 3. Run REGENIE; comment accordingly for model-fitting or testing on imputed genotypes
then 
    module load Boost/1.71.0-gompic-2019b
    module load intel/2020a
    tag=regenie
    $REGENIE --step 1 --bed $geno --phenoFile $pheno --covarFile $covar --bsize 1000 --threads 12 --lowmem --lowmem-prefix $work_dir/regenie_tmp --out $work_dir/$tag --gz
    $REGENIE --step 2 --bed $geno --phenoFile $pheno --covarFile $covar --bsize 500 --threads 12 --minMAC 1 --pred $work_dir/$tag\_pred.list --out $work_dir/$tag --gz
   for C in {1..22}; do
       tag=imp.regenie
       echo -e "#!/bin/bash\n module load GCCcore/8.3.0\n module load Boost/1.71.0-gompic-2019b\n module load intel/2020a" > temp.reg
       BGEN=/FIXTHIS/ukb50k_imp_chr$C
       echo "$REGENIE --step 2 --bgen $BGEN.bgen --sample $BGEN.sample --keep $annoPrefix.samples --ref-first --minMAC 5 --minINFO 0.50 --phenoFile $work_dir/phenotypes.tab --covarFile $covar --bsize 400 --threads 12 --pred $work_dir/regenie.all\_pred.list --out $work_dir/imputed/imp.chr$C.regenie --gz" >> temp.reg
       qsub -N real20.$tag.$2.chr$C -pe shmem 12 -q short.qc@@short.hge -cwd temp.reg
   done
fi

if [ $1 = "bolt" ]; # 4. Run BOLT on those; remember to add the commands for testing on imputed genotypes
then
	cd $work_dir
 	for pheno in 'BMI' 'MeanPTV' 'ForcedVitalCapacity' 'SystolicBloodPressure' 'DiastolicBloodPressure' 'RedBloodCellCount' 'MeanCorpVol' 'MeanCorpHaem' 'RedBloodCellDW' 'WhiteBloodCellCount' 'PlateletCount' 'PlateletCrit' 'PlateletDW' 'MonocyteCount' 'EosinoCount' 'EosinoPercentage' 'MeanSCV' 'TotalCholesterol' 'HDLCholesterol' 'GlycatedHaemoglobinHbA1c';
	do
		echo "Applying Bolt to $pheno" #--covarCol=Site --covarMaxLevels=30 
		echo "$BOLT --bfile $geno --phenoFile=phenotypes.tab --phenoCol=$pheno --covarFile=covariates.tab --covarCol=Sex --qCovarCol=Age --qCovarCol=Age2 --qCovarCol=PCA{1:20} --lmmForceNonInf --LDscoresFile=$ldPrefix.score.ld --LDscoresCol ldscore --verboseStats --statsFile=bolt.$pheno.sumstats.gz --numThreads=12" > bolt.temp
        qsub -N real20.bolt.$2.$pheno -pe shmem 12 -q long.qc@@long.hge -cwd bolt.temp
		# --bgenFile=/FIXTHIS/imputation/ukb_imp_chr{1..22}_v3.bgen --bgenMinMAF=1e-5 --bgenMinINFO=0.5 --sampleFile=/FIXTHIS/.sample --remove bolt.in_plink_but_not_imputed.FID_IID.txt --remove bolt.remove_these.txt --statsFileBgenSnps=imputed/imp.bolt.BMI.gz --verboseStats
	done
fi

if [ $1 = "fma_imp" ]; # 5. Run LINEAR REGRESSION on imputed data; we can either run it on the raw pheno with covariates, or at the residualised phenotype
then # NOTE: I think it would be better to first create a file with the LOCO residuals for all traits and one chromosome, then run plink on that chrom (instead of loading the same chrom again and again)
# 1) prepare residuals, 2) run PLINK, 3a) grep ADD lines (if with covariates) and gzip, 3b) convert to chisqr and calibrate  
#     python3 fix_plink.py $2 loco rctamc.C16 # better run this independently
    for C in {1..22}; do
        tag="C8"
        echo -e "#!/bin/bash\nmodule load Python/3.7.4-GCCcore-8.3.0" > temp.rcta
        echo "$PLINK --pgen $PGEN\_chr$C\_v3.pgen --psam $PGEN\_chr$C\_v3.psam --pvar $PGEN\_chr$C\_v3.pvar --keep-fam $annoPrefix.samples --mac 5 --mach-r2-filter 0.50 2 --glm hide-covar --pheno $work_dir/rctamc.C8.loco$C.residuals.gz --threads 12 --memory 85000 --out $work_dir/imputed/imp.chr$C.$tag" >> temp.rcta #--hwe 1e-7
        echo "python3 ../src/fix_plink.py stats $work_dir/imputed $tag $C results_ukbb_$2/rctamc.C8.calibrators $tag" >> temp.rcta
        echo "rm $work_dir/imputed/imp.chr$C.$tag.*linear" >> temp.rcta
        qsub -N real20.imp.$tag.chr$C -pe shmem 12 -q short.qc@@short.hge -cwd temp.rcta
    done
#     rm temp.rcta
fi

if [ $1 = "fastgwa" ];
then
    
    if ! [ -e results_446k/ukbb.446k.grm.sp ]
    then
        echo "Calculate the GRM based on the bed file"
        prefix=results_446k/ukbb.446k.grm.sp
        for i in {1..10}; do
            $GCTA --bfile $geno --make-grm-part 10 $i --thread-num 5 --out $prefix
        done
        cat $prefix.part_10_*.grm.id > $prefix.grm.id
        cat $prefix.part_10_*.grm.bin > $prefix.grm.bin
        cat $prefix.part_10_*.grm.N.bin > $prefix.grm.N.bin
        $GCTA --grm $prefix --make-bK-sparse 0.05 --out $prefix --thread-num 12
        rm $prefix*part*
    fi
    cd $work_dir
    for ((i=0; i<20; i++))
	do
#         tag=fastgwa_nocov
#         pheno=$(awk -v c=$(($i+3)) 'NR==1{print $c}' rctamc.C8.adjusted_traits.phen)
#         echo "Applying fastGWA-NoCov to $pheno using $2"
#         awk -v c=$(($i+3)) 'NR>1{print $1, $2, $c}' rctamc.C8.adjusted_traits.phen > $tag.$pheno.txt # GCTA needs no header
#         echo "$GCTA --grm-sparse ../results_ukbb_446k/ukbb.446k --fastGWA-mlm --pheno $tag.$pheno.txt --out imputed/imp.$tag.$pheno --mbgen ../bgen.all_chroms.txt --sample /well/palamara/projects/UKBB_APPLICATION_43206/new_copy/data_download/ukb22828_c22_b0_v3_s487276.sample --extract snplist.imputed.txt --maf 0.00001" > $tag.temp 
#         --geno 0.5 # --threads 12 
# --info 0.5
#         echo "$GCTA --mpfile ../pgen.all_chroms.txt --grm-sparse ../results_ukbb_446k/ukbb.446k --fastGWA-mlm --pheno $tag.$pheno.txt --threads 12 --out imputed/imp.$tag.$pheno --maf 0.00001 --extract snplist.imputed.txt" > $tag.temp 
         
        tag=fastgwa
        pheno=$(awk -v c=$(($i+3)) 'NR==1{print $c}' phenotypes.tab)
        echo "Applying fastGWA to $pheno using $2"
        awk -v c=$(($i+3)) 'NR>1{print $1, $2, $c}' phenotypes.tab > $tag.$pheno.txt # GCTA needs no header
#         echo "$GCTA --bfile $geno --grm-sparse ../results_ukbb_446k/ukbb.446k --fastGWA-mlm --pheno $tag.$pheno.txt --qcovar covariates.tab --threads 12 --out fastgwa.$pheno" > $tag.temp
        echo "$GCTA --mpfile pgen.all_chroms.txt --grm-sparse ../results_ukbb_446k/ukbb.446k --fastGWA-mlm --pheno $tag.$pheno.txt --qcovar covariates.tab --threads 12 --out imputed/imp.$tag.$pheno --maf 0.0001" > $tag.temp  #--extract snplist.imputed.$2.txt
         
#         echo "gzip -f imputed/imp.$tag.$pheno.fastGWA" >> $tag.temp
        echo "rm $tag.$pheno.txt" >> $tag.temp
        . $tag.temp
#         qsub -N real20.$tag.$2.$pheno -pe shmem 12 -q short.qc@@short.hge -cwd $tag.temp -P palamara.prjc.high
	done
fi

if [ $1 = "linreg" ]; # 6. Run LINEAR REGRESSION on genotyped/imputed data
then 
    for C in {1..22}; do
        echo -e "#!/bin/bash" > temp.plink
        # echo "$PLINK --pgen $GENOPATH/ukb_imp_chr$C\_v3.pgen --psam $GENOPATH/ukb_imp_chr$C\_v3.psam --pvar $GENOPATH/ukb_imp_chr$C\_v3.pvar --keep-fam $annoPrefix.samples --glm hide-covar --pheno $work_dir/rctamc.C16.adjusted_traits.phen --mach-r2-filter 0.50 2 --mac 5 --threads 12 --memory 8192 --out $work_dir/imputed/imp.chr$C.linreg" >> temp.plink
        echo "$PLINK --pgen $PGEN\_chr$C\_v3.pgen --psam $PGEN\_chr$C\_v3.psam --pvar $PGEN\_chr$C\_v3.pvar --keep-fam $annoPrefix.samples --glm  hide-covar --pheno $work_dir/phenotypes.tab --covar $work_dir/covariates.tab --covar-col-nums 3,4,6-25 --covar-variance-standardize --mach-r2-filter 0.50 2 --mac 5 --threads 12 --memory 15360 --out $work_dir/imputed/imp.chr$C.linreg" >> temp.plink
        echo "for x in $work_dir/imputed/imp.chr$C.linreg.*linear; do gzip \$x; done" >> temp.plink 
        qsub -P palamara.prjc.high -N real20.imp.linreg.chr$C -pe shmem 12 -q short.qc@@short.hge -cwd temp.plink
    done
fi

date

# submit tasks like
# method="fma"; case=446k; qsub -N real20.$method.$case -pe shmem 12 -q short.qc@@short.hge -cwd runthis_real20.sh $method $case
