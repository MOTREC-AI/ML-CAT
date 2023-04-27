
library(cmprsk)
library(prodlim)
library(mltools)
library(riskRegression)
library(readr)
library(randomForestSRC)
library(ROCR)
library(tictoc)
library(cvAUC)
library(intsurv)
library(survival)
library(scales)
library(wrapr)
library(data.table)
library(dplyr)

print("Starting...")
Sys.time()

print("randomForestSRC version:")
packageVersion("randomForestSRC")

# Turn on both OpenMP and R-side parrallel processing.
library("parallel")
options(rf.cores=4, mc.cores = 4)
#################################################

args <- commandArgs(TRUE)
LSB_JOBINDEX <- as.integer(args[1])
dataDir <- args[2]               
train <- args[3]
train_y <- args[4]
val <- args[5]
val_y <- args[6]
outPrefix <- args[7]
mtry <- as.integer(args[8])
nodesize <- as.integer(args[9])
ntrees <- as.integer(args[10])

#################################################
# We use Python back-end to compute Antolini C-index
Sys.setenv(RETICULATE_PYTHON = <python PATH>)
library("reticulate")

# Load python packages
pycox <- import("pycox")
pd <- import("pandas")
skl <- import("sklearn.model_selection")
nb <- import("numbers")

#################################################

source("comparison_utility_functions.R")

print("starting to fit models...")
print(paste0("Array ID is: ", LSB_JOBINDEX))

# Fitting models - choose rsf or fg

# rsf:
# output_rsf <- comparator_py(train, train_y, val, val_y, algo = "rsf", dataDir, outPrefix, LSB_JOBINDEX, mtry, nodesize, ntrees)

# fg:
output_fg <- comparator_py(train, train_y, val, val_y, algo = "fg", dataDir, outPrefix, LSB_JOBINDEX, mtry, nodesize, ntrees)
