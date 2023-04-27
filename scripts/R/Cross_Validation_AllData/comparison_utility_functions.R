tester <- function(train_kf, train_kf_y, val_kf, val_kf_y, algo, mtry, nodesize, ntrees)
{
  train_data <- as.data.frame(fread(train_kf))     
  train_yVals <- as.data.frame(fread(train_kf_y))
  val_data <- as.data.frame(fread(val_kf))      
  val_yVals <- as.data.frame(fread(val_kf_y))

  ######################################################################################################
  # Kluge to get rid of CMP features when required for a covariate set. Comment out otherwise.
#     train_data$SODIUM <- NULL
#     train_data$POTASSIUM <- NULL
#     train_data$CHLORIDE <- NULL
#     train_data$CALCIUM	<- NULL
#     train_data$CO2 <- NULL
#     train_data$GLUCOSE	<- NULL
#     train_data$UREA <- NULL
#     train_data$CREATININE <- NULL
#     train_data$TPROTEIN <- NULL
#     train_data$AST <- NULL
#     train_data$ALT <- NULL
#     train_data$TBLI <- NULL
#     train_data$ALKPHOS <- NULL
#   
#     val_data$SODIUM <- NULL
#     val_data$POTASSIUM <- NULL
#     val_data$CHLORIDE <- NULL
#     val_data$CALCIUM	<- NULL
#     val_data$CO2 <- NULL
#     val_data$GLUCOSE	<- NULL
#     val_data$UREA <- NULL
#     val_data$CREATININE <- NULL
#     val_data$TPROTEIN <- NULL
#     val_data$AST <- NULL
#     val_data$ALT <- NULL
#     val_data$TBLI <- NULL
#     val_data$ALKPHOS <- NULL
  ######################################################################################################

  covars <- colnames(train_data)

  ######################################################################################################
  # Kluge to do the "Basic Genetic" & "Basic Binarized" covariate sets. It's "No Labs" & "Simple", with the following adjustment (comment out otherwise):
#   chemo_pos <- grepl("CHEMO_", covars)
# 
#   train_data[,chemo_pos] <- ifelse(train_data[,chemo_pos] < 28, 1, 0)
#   hasChemo <- rowSums(train_data[,chemo_pos])
#   hasChemo <- ifelse(hasChemo > 0, 1, 0)
#   train_data[,chemo_pos] <- list(NULL)
#   train_data$hasChemo <- hasChemo
# 
#   val_data[,chemo_pos] <- ifelse(val_data[,chemo_pos] < 28, 1, 0)
#   hasChemo <- rowSums(val_data[,chemo_pos])
#   hasChemo <- ifelse(hasChemo > 0, 1, 0)
#   val_data[,chemo_pos] <- list(NULL)
#   val_data$hasChemo <- hasChemo
# 
#   train_data$DX_delta <- NULL
#   val_data$DX_delta <- NULL
# 
#   covars <- colnames(train_data)
  ######################################################################################################

  data <- cbind(train_yVals, train_data)

  model <- model_maker(data, covars, algo, mtry, nodesize, ntrees)

  data <- cbind(val_yVals, val_data)
  predictions_test <- predictor(model, data, covars, algo)

  # Cleaning up (RSF models can be large)
  rm(model)
  gc()
  
  out <- list()
  out$outcomes_vte_test <- data$EVENT_6
  out$outcomes_vte_test[out$outcomes_vte_test==2] <- 0
  out$outcomes_death_test <- data$EVENT_6
  out$outcomes_death_test[out$outcomes_death_test==1] <- 0
  out$outcomes_death_test[out$outcomes_death_test==2] <- 1
  
  # Old Harrell c-index (in intsurv)
  if (algo == "rsf") 
  {
     # rsf
     out$c_index_vte <- cIndex(time = data$OBS_TIME_6, event = data$EVENT_6, risk_score = predictions_test$predicted$predicted[,1], weight = NULL)[1]   
  }else
  {
     # fg
     out$c_index_vte <- cIndex(time = data$OBS_TIME_6, event = data$EVENT_6, risk_score = predictions_test$predicted_vte[,1], weight = NULL)[1]
  }
  out$c_index_death <- NA

  # C-stat calculations
  out$c_index_vte2 <- c_index(predictions_test$predicted_vte, data$OBS_TIME_6, out$outcomes_vte_test)

  if (!is.null(dim(predictions_test$predicted_death)))
  {
    out$c_index_death <- c_index(predictions_test$predicted_death, data$OBS_TIME_6, out$outcomes_death_test)
  }
  
  out
  
}


# Fits a batch of models and evaluates on the training set using k-fold cross-validation, returns  C-indexes
batcher_kf <- function(train, train_y, val, val_y, algo, dataDir, outPrefix, LSB_JOBINDEX, mtry, nodesize, ntrees)
{
  print("starting batcher...")
  i <- LSB_JOBINDEX
  set.seed(1 + i)

  print(paste0("In batcher_kf, i is: ", i))
      
  train_kf <- paste0(dataDir, train, "_", i, ".csv")
  train_kf_y <- paste0(dataDir, train_y, "_", i, ".csv")
  val_kf <- paste0(dataDir, val, "_", i, ".csv")
  val_kf_y <- paste0(dataDir, val_y, "_", i, ".csv")

  output_kf <- tester(train_kf, train_kf_y, val_kf, val_kf_y, algo, mtry, nodesize, ntrees)

  cindex_vte_test <- output_kf$c_index_vte   # Harrel
  cindex_vte_test2 <- output_kf$c_index_vte2 # Antolini

  if (algo == "rsf") cindex_death_test <- output_kf$c_index_death else cindex_death_test <- NA
      
  print(paste0("cindex_vte_test is: ", cindex_vte_test))
  print(paste0("cindex_vte_test2 is: ", cindex_vte_test2))
  print(paste0("cindex_death_test is: ", cindex_death_test))

  outfile <- paste0("Out/", outPrefix, "_", i, ".csv")
  outdata <- data.frame(cindex_vte_test = c(cindex_vte_test), cindex_vte_test2 = c(cindex_vte_test2), cindex_death_test = c(cindex_death_test))
  fwrite(outdata, file = outfile, append = FALSE)

  print("leaving batcher")
}


# Fits one model
model_maker <- function(data, covars, algo, mtry, nodesize, ntrees)
{
  if (algo == "rsf") {
    
    formula <- paste("Surv(OBS_TIME, EVENT)~", paste(covars, collapse="+"), sep="")
    formula <- as.formula(formula)
    
    # Find position of first genetic covariate
    alt_pos <- grepl("_alt", covars)

    split.weight <- c(rep(1, length(covars)))

    data <- subset(data, select = c(covars, "OBS_TIME", "EVENT"))

    model <- rfsrc(formula, data = data, na.action = "na.impute", splitrule = "logrank", split.wt = split.weight, mtry = mtry, nodesize = nodesize, ntree = ntrees, importance = FALSE, block.size = "ntree")
  } else if (algo == "fg") {
    
    formula <- paste0("Hist(OBS_TIME,EVENT)~", paste(covars, collapse = "+"))
    formula <- as.formula(formula)

    model <- FGR(formula, data, cause = 1, variance = FALSE)
  }
  
  model
}


# Gives cumulative incidence function of and event at 180 days for patients of the test set
predictor <- function(model, data, covars, algo)
{
  out <- list()
  
  if (algo == "rsf") {
    
    predicted <- predict(model, data, na.action = "na.omit")

    times <- predicted$time.interest  # This is a vector of all the times for which a cif has been calculated.

    time.slot <- if (length(which(times == 180)) == 1) which(times == 180) else sum(times < 180)  # Find the index of the time slot for time <=180
    predictions.180 <- predicted$cif[,time.slot,1]  # In cif there is one row for each sample.  The third index is for the 2 risk factors.
    out$predicted <- predicted
    out$predictions.180 <- predictions.180  # This is a vector of cif's at time<=180 for each sample

    out$predicted_vte <- cif_standardizer(predicted$cif[,,1], times)
    out$predicted_death <- cif_standardizer(predicted$cif[,,2], times)

  } else if (algo == "fg") {
    
    cif <- predict(model, data)
    time_points <- model$crrFit$uftime

    out$predicted_vte <- cif_standardizer(cif, time_points)
    out$predicted_death <- NA
  }
  
  out
}


# Standardize CIF matrix to have all of 180 days
cif_standardizer <- function(cif_mat, time_points){
  
  time_points <- time_points[time_points<=180]
  cif_mat <- cif_mat[, 1:length(time_points)]
  new_mat <- matrix(nrow = nrow(cif_mat), ncol = 180)
  new_mat[,time_points] <- cif_mat
  
  # If day #1 not assessed we assign cumulative incidence = 0
  if (!(1 %in% time_points)) new_mat[,1] <- 0
  
  # For days not assessed we assign cumulative incidence of prior day
  for (i in 2:180){
    
    if (!(i %in% time_points)) new_mat[,i] <- new_mat[, i-1]
    
  }
  
  new_mat
}


# Compares multiple models (actually, just 1 for now)
comparator_py <- function(train, train_y, val, val_y, algo, dataDir, outPrefix, LSB_JOBINDEX, mtry, nodesize, ntrees)
{
  print("in comparator")

  batcher_kf(train, train_y, val, val_y, algo, dataDir, outPrefix, LSB_JOBINDEX, mtry, nodesize, ntrees)
}


# Computes C-index using Antolini method (Python execution)
c_index <- function(cif, times, events){
  
  t_cif <- t(cif)
  cif_df <- pd$DataFrame(1-t_cif)
  
  times <- np_array(times, dtype = "int64", order = "C")
  events <- np_array(events, dtype = "int64", order = "C")
  
  results <- pycox$evaluation$EvalSurv(cif_df, times, events)
  out <- results$concordance_td('antolini')
}


# Tabulating results
tabulator <- function(output){
  
  covar_list <- output$covar
  covar_len <-length(covar_list)
  results <- data.frame(covars = unlist(lapply(covar_list, paste, collapse=", ")), cindex_vte_test = rep(NA, covar_len), 
                        cindex_vte_test_SE = rep(NA, covar_len), cindex_vte_test_CI = rep(NA, covar_len),
                        cindex_death_test = rep(NA, covar_len))

  for (i in 1:covar_len){
  
    results_vte <- unlist(output$results[[i]]$cindex_vte_test)
    results$cindex_vte_test[i] <- round(mean(results_vte), digits = 4)
    results$cindex_vte_test_SE[i] <- round(sd(results_vte)/sqrt(length(results_vte)), digits = 4)
    results$cindex_vte_test_CI[i] <- paste(round(results$cindex_vte_test[i]-1.96*results$cindex_vte_test_SE[i], digits = 4), 
                                           "-",
                                           round(results$cindex_vte_test[i]+1.96*results$cindex_vte_test_SE[i], digits = 4))
    if (!is.null(unlist(output$results[[i]]$cindex_death_test)[1])) results$cindex_death_test[i] <- round(mean(unlist(output$results[[i]]$cindex_death_test)), digits = 4)
    
  }
  
  results
}

