#!/bin/bash
####################################################
# This script launches the cross validation runs for data sets containing data from all patients
# It uses IBM's LSF, R, and python

nRuns=20       # Number of validation folds
dataDir= <PATH to directory with data>

# Prefixes to validation fold data files
train_y="train_y"
val_y="val_y"


# Select a train_name and a val_name from the lines below. Each corresponds to a different set of covariates
# train_name="ext_train_features"
# val_name="ext_val_features"
# train_name="main_train_features"
# val_name="main_val_features"
# train_name="simple_train_features"
# val_name="simple_val_features"
# train_name="genes_only_train_features"
# val_name="genes_only_val_features"
# train_name="genes_cancer_type_train_features"
# val_name="genes_cancer_type_val_features"
# train_name="no_labs_train_features"
# val_name="no_labs_val_features"
train_name="no_genes_train_features"
val_name="no_genes_val_features"
# train_name="cancer_type_only_train_features"
# val_name="cancer_type_only_val_features"

# Select hyperparameters for the random survival forest code.  mtry and nodesize can be set to loop over a ranger of values. See the for loops below.
mtry_start=2
mtry_end=2
nodesize_start=38
nodesize_end=38

ntrees=500 # Setting ntrees higher than 500 provided very small improvements at high compute cost.

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
                           outPrefix=${outPrefix}, mtry=${mtry}, nodesize=${nodesize}, ntrees=${ntrees}" \
                     < run_arrays.bsub)                            
        
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

