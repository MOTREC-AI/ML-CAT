#!/bin/bash
####################################################
# This script launches the cross validation runs for data sets containing data from only KS patients
# It uses IBM's LSF, R, and python

nRuns=20       # Number of validation folds
dataDir= <PATH to directory with data>

# Prefixes to validation fold data files
train_y="train_y"
val_y="val_y"

# Select a train_name, a val_name, a ks_train_TrueFalse, and a ks_val_TrueFalse from the lines below. Each corresponds to a different set of covariates
# train_name="ext_train_features"
# val_name="ext_val_features"
# ks_train_TrueFalse="ext_ks_train_TrueFalse"
# ks_val_TrueFalse="ext_ks_val_TrueFalse"
# 
# train_name="main_train_features"
# val_name="main_val_features"
# ks_train_TrueFalse="main_ks_train_TrueFalse"
# ks_val_TrueFalse="main_ks_val_TrueFalse"
# 
train_name="simple_train_features"
val_name="simple_val_features"
ks_train_TrueFalse="simple_ks_train_TrueFalse"
ks_val_TrueFalse="simple_ks_val_TrueFalse"
# 
# train_name="genes_only_train_features"
# val_name="genes_only_val_features"
# ks_train_TrueFalse="genes_only_ks_train_TrueFalse"
# ks_val_TrueFalse="genes_only_ks_val_TrueFalse"
# 
# train_name="genes_cancer_type_train_features"
# val_name="genes_cancer_type_val_features"
# ks_train_TrueFalse="genes_cancer_type_ks_train_TrueFalse"
# ks_val_TrueFalse="genes_cancer_type_ks_val_TrueFalse"
# 
# train_name="no_labs_train_features"
# val_name="no_labs_val_features"
# ks_train_TrueFalse="no_labs_ks_train_TrueFalse"
# ks_val_TrueFalse="no_labs_ks_val_TrueFalse"
# 
# train_name="no_genes_train_features"
# val_name="no_genes_val_features"
# ks_train_TrueFalse="no_genes_ks_train_TrueFalse"
# ks_val_TrueFalse="no_genes_ks_val_TrueFalse"
# 
# train_name="cancer_type_only_train_features"
# val_name="cancer_type_only_val_features"
# ks_train_TrueFalse="cancer_type_only_ks_train_TrueFalse"
# ks_val_TrueFalse="cancer_type_only_ks_val_TrueFalse"

# Select hyperparameters for the random survival forest code.  mtry and nodesize can be set to loop over a ranger of values. See the for loops below.
mtry_start=2
mtry_end=2
nodesize_start=62
nodesize_end=62

ntrees=500

outPrefixBase="N500"  # Setting ntrees higher than 500 provided very small improvements at high compute cost.

# Prefix to the output files found in the "Out" directory
outPrefixBase="N500"
#########################################################################################################
# Launch LSF jobs

for (( mtry=mtry_start; mtry<=mtry_end; ((mtry+=2)) ))
{
   for (( nodesize=nodesize_start; nodesize<=nodesize_end; ((nodesize+=2)) ))
   {
       outPrefix="${outPrefixBase}_${mtry}_${nodesize}_mtry_nodesize"
        
       JOB_ID_TEXT=$(bsub -J rsf_lsf[1-"${nRuns}"]%20 -n 4 \
                     -env "all, dataDir=${dataDir}, train_name=${train_name}, train_y=${train_y}, val_name=${val_name}, val_y=${val_y}, \
                           ks_train_TrueFalse=${ks_train_TrueFalse}, ks_val_TrueFalse=${ks_val_TrueFalse}, \
                           outPrefix=${outPrefix}, mtry=${mtry}, nodesize=${nodesize}, ntrees=${ntrees}" \
                     < run_arrays_ks.bsub)                           
        
       echo "JOB_ID_TEXT is: ${JOB_ID_TEXT}"
       JOB_ID1=${JOB_ID_TEXT#*Job <}
       JOB_ID1=${JOB_ID1%> is*}
       echo "JOB_ID1 is: ${JOB_ID1}"
        
        
       # Collect output from array jobs, and summarize.
       JOB_ID_TEXT=$(bsub -w "done(${JOB_ID1})" -J CRsumry -n 1 \
                     -env "all, nRuns=${nRuns}, outPrefix=${outPrefix}" \
                     < array_summarize.bsub)
        
       echo "JOB_ID_TEXT is: ${JOB_ID_TEXT}"
       JOB_ID2=${JOB_ID_TEXT#*Job <}
       JOB_ID2=${JOB_ID2%> is*}
       echo "JOB_ID2 is: ${JOB_ID2}"
    }
}

# Check on the status of your running jobs. "bjobs -a" also shows recently completed jobs.
bjobs

