#!/bin/bash
####################################################
# This script launches the runs for Test data sets containing data from all patients
# It uses IBM's LSF, R, and python

dataDir= <PATH to directory with data>

# Prefixes to full data files
train_y="full_train_y_imputed_iterative.csv"
val_y="full_test_y_imputed_iterative.csv"

# Select a train_name and a val_name from the lines below. Each corresponds to a different set of covariates (val is really the test data set here)
# train_name="ext_train_features"
# val_name="ext_test_features"
# train_name="main_train_features"
# val_name="main_test_features"
# train_name="simple_train_features"
# val_name="simple_test_features"
# train_name="genes_only_train_features"
# val_name="genes_only_test_features"
# train_name="genes_cancer_type_train_features"
# val_name="genes_cancer_type_test_features"
# train_name="no_labs_train_features"
# val_name="no_labs_test_features"
train_name="no_genes_train_features"
val_name="no_genes_test_features"
# train_name="cancer_type_only_train_features"
# val_name="cancer_type_only_test_features"

train_name="${train_name}_full_imputed_iterative.csv"
val_name="${val_name}_full_imputed_iterative.csv"

# Set the random survival forest hyperparameters for this covariate set
mtry=2
nodesize=38
ntrees=500   # Setting ntrees higher than 500 provided very small improvements at high compute cost.
nRuns=1

rm -f Out/testing.csv

#########################################################################################################
   
JOB_ID_TEXT=$(bsub -J rsf_lsf[1-"${nRuns}"]%20 -n 4 \
              -env "all, dataDir=${dataDir}, train_name=${train_name}, train_y=${train_y}, val_name=${val_name}, val_y=${val_y}, \
                    mtry=${mtry}, nodesize=${nodesize}, ntrees=${ntrees}" \
              < run_arrays_test2.bsub)                            
        
echo "JOB_ID_TEXT is: ${JOB_ID_TEXT}"
JOB_ID1=${JOB_ID_TEXT#*Job <}
JOB_ID1=${JOB_ID1%> is*}
echo "JOB_ID1 is: ${JOB_ID1}"
        
# Check on the status of your running jobs. "bjobs -a" also shows recently completed jobs.
bjobs

