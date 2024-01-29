#!/bin/bash

### OVERVIEW ###
# Assess scalability in h2 estimation
# 0. Use a jupyter notebook to create the lists (UWB, RWB, EUR) of samples (as for association)
# 1. apply rhe, GCTA-HE, BOLT-REML, on those using `runthis_h2.sh rhe/rhemt/bolt/gcta/prepare 10k/25k/50k`
# 2. analyse!

### WARNINGS ### 
# 1. The genotype/sample QC is not included here
# 2. This script is designed for clusters with an SGE scheduler (ie `qsub`); please modify accordingly 
# 3. Remember to fix the ld-scores file:
#   sed -i '$d' ../PREFIX.score.ld
#   sed -i 's/ldscore/LDSCORE/g' ../$prefix.score.ld (needed for BOLT-LMM)
# 4. Need to exclude the HLA region from the annotation for RHE to work more accurately

date

N=$2
design=UWB_a075b1
work_dir=/FIXTHIS/assess_h2
genopath=/FIXTHIS/DATASETS
genoprefix=$genopath/ukbb.UWB_$N
pheno=$work_dir/pheno_$N.tab

## requirements
source /FIXTHIS/mypython-${CPU_ARCHITECTURE}/bin/activate
module load Python/3.7.4-GCCcore-8.3.0
BOLT=/FIXTHIS/BOLT-LMM_v2.3.4/bolt
REGENIE=/FIXTHIS/regenie-3.0.1/regenie
RHEv1=/FIXTHIS/RHE-mc_single/build/RHEmc
RHEMT=/FIXTHIS/RHE-mt-new/build/mtRHEmc
PLINK=/FIXTHIS/plink2
GCTA=/FIXTHIS/gcta_1.93.2beta/gcta64

repeats=50
nThreads=1

if [ $1 = "prepare" ]; then
	
	# make a new bed file
	$PLINK --bfile $genopath/ukbb.UWB_50k --keep $work_dir/samples.$N.txt --make-bed --out $genopath/ukbb.UWB_$N --memory 15000 --threads $nThreads
	
	if ! [ -e $genopath/ukbb.UWB_$N.grm.bin ]
	then
		# create the grms for GCTA
		echo "Creating a GRM with GCTA"
		$GCTA --bfile $genoprefix --make-grm-part 3 1 --thread-num 5 --out $genoprefix
		$GCTA --bfile $genoprefix --make-grm-part 3 2 --thread-num 5 --out $genoprefix
		$GCTA --bfile $genoprefix --make-grm-part 3 3 --thread-num 5 --out $genoprefix
		cat $genoprefix.part_3_*.grm.id > $genoprefix.grm.id
		cat $genoprefix.part_3_*.grm.bin > $genoprefix.grm.bin
		cat $genoprefix.part_3_*.grm.N.bin > $genoprefix.grm.N.bin
		cat $genoprefix.part_3_*.log > $genoprefix.grm.log
		rm $genoprefix.part_3_*
	fi

elif [ $1 = "bolt" ]; then
	echo "run Bolts on $design-$N"
	cd $work_dir
	for (( t=0; t<$repeats; t++ ));
	do
		echo "$BOLT --reml --remlNoRefine --bfile $genoprefix --phenoFile=$pheno --phenoCol=pheno$t --numThreads=$nThreads" > bolt.temp
	   #--covarFile=Covariates_all.tab --qCovarCol=PCA{1:5}--modelSnps $genoprefix.maf2_ld4.bolt
	   qsub -N bolt.$N.$t -pe shmem $nThreads -q short.qc@@short.hge -cwd bolt.temp
	done
	cd ../
	
elif [ $1 = "gcta" ]; then
	echo "run GCTA on $design-$N"
	for (( t=1; t<=$repeats; t++ ));
	do
		echo $N, $t
  		$GCTA --grm $genoprefix --HEreg --pheno $pheno --mpheno $t --out assess_h2/gcta.$N.$t --thread-num $nThreads
     	$GCTA --grm $genoprefix --reml  --pheno $pheno --mpheno $t --grm-adj 0 --out assess_h2/gcta.$N.$t --thread-num $nThreads
	done	
	
elif [ $1 = "rhemt" ]; then
	echo "run RHE-mt on $design-$N"
	
 	$RHEMT -g $genoprefix -p $pheno -jn 10 -o $work_dir/rhe.$N -annot $genopath/ukbb.UWB_50k.uniform.annot -k 50
# 	$RHEMT -g $genoprefix -p $pheno -jn 10 -o $work_dir/rhe.C8.$N -annot $genopath/ukbb.UWB_50k.maf2_ld4.annot -k 50
#  	$RHEMT -g $genoprefix -p $pheno -jn 10 -o $work_dir/rhe.C16.$N -annot $genopath/ukbb.UWB_50k.maf4_ld4.annot -k 50
    
    echo "Finished!"
    date
elif [ $1 = "rhe" ]; then
	for (( t=1; t<=$repeats; t++ ));
	do
		echo "run RHE (old - one trait) on $design-$N-$t"
		awk -v c=$(($t+2)) '{print $1, $2, $c}' $pheno > $pheno.$N.$t.tmp
		$RHEv1 -g $genoprefix -p $pheno.$N.$t.tmp -jn 10 -o $work_dir/rheold.$N.$t -annot $genopath/ukbb.UWB_50k.uniform.annot -k 10
		rm $pheno.$N.$t.tmp
	done
	
	echo "Finished!"
else
	echo "Unrecognized command: $1, $2"
fi
date
# subimt as
# method="rhemt"; for case in 10k 25k 50k; do qsub -N h2task.$method.$case -pe shmem 5 -q short.qc@@short.hge -cwd runthis_h2.sh $method $case; done
