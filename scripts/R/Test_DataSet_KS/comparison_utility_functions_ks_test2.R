tester <- function(train_kf, train_kf_y, val_kf, val_kf_y, algo, mtry, nodesize, ntrees)
{
  train_data <- as.data.frame(fread(train_kf))     
  train_yVals <- as.data.frame(fread(train_kf_y))
  val_data <- as.data.frame(fread(val_kf))      
  val_yVals <- as.data.frame(fread(val_kf_y))

  # Find the KS patients
  xFile <- as.data.frame(fread("<PATH to>/full_test_x_imputed_iterative.csv"))
  ksVec <- !is.na(xFile$KS)

  colnames(val_data) <- str_replace(colnames(val_data), "_ks", "")

  val_data <- val_data[ksVec,]
  val_yVals <- val_yVals[ksVec,]

  # Fix the "# AGE" problem
  colOnename <- colnames(train_data)[1]
  colOnename <- str_replace(colOnename, "# ", "")
  colnames(train_data)[1] <- colOnename

  colOnename <- colnames(val_data)[1]
  colOnename <- str_replace(colOnename, "# ", "")
  colnames(val_data)[1] <- colOnename

  # PHI adjustment
  if("AGE" %in% colnames(train_data))
  {
     train_data$AGE[train_data$AGE > 90] <- 90
  }

  if("AGE" %in% colnames(val_data))
  {
     val_data$AGE[val_data$AGE > 90] <- 90
  }

all_event_data <- train_yVals
all_event_data2 <- all_event_data
colnames(all_event_data) <- c("EVENT", "OBS_TIME")
colnames(all_event_data2) <- c("EVENT_6", "OBS_TIME_6") # We'll populate these in the next four lines.
all_event_data2$OBS_TIME_6[all_event_data$OBS_TIME > 180] <- 180
all_event_data2$EVENT_6[all_event_data$OBS_TIME > 180] <- 0
train_yVals <- cbind(all_event_data, all_event_data2)

all_event_data <- val_yVals
all_event_data2 <- all_event_data
colnames(all_event_data) <- c("EVENT", "OBS_TIME")
colnames(all_event_data2) <- c("EVENT_6", "OBS_TIME_6") # We'll populate these in the next four lines.
all_event_data2$OBS_TIME_6[all_event_data$OBS_TIME > 180] <- 180
all_event_data2$EVENT_6[all_event_data$OBS_TIME > 180] <- 0
val_yVals <- cbind(all_event_data, all_event_data2)

  ######################################################################################################
  # Kluge to get rid of CMP features when required for a covariate set. Comment out otherwise.
  train_data$SODIUM <- NULL
  train_data$POTASSIUM <- NULL
  train_data$CHLORIDE <- NULL
  train_data$CALCIUM	<- NULL
  train_data$CO2 <- NULL
  train_data$GLUCOSE	<- NULL
  train_data$UREA <- NULL
  train_data$CREATININE <- NULL
  train_data$TPROTEIN <- NULL
  train_data$AST <- NULL
  train_data$ALT <- NULL
  train_data$TBLI <- NULL
  train_data$ALKPHOS <- NULL

  val_data$SODIUM <- NULL
  val_data$POTASSIUM <- NULL
  val_data$CHLORIDE <- NULL
  val_data$CALCIUM	<- NULL
  val_data$CO2 <- NULL
  val_data$GLUCOSE	<- NULL
  val_data$UREA <- NULL
  val_data$CREATININE <- NULL
  val_data$TPROTEIN <- NULL
  val_data$AST <- NULL
  val_data$ALT <- NULL
  val_data$TBLI <- NULL
  val_data$ALKPHOS <- NULL
  ######################################################################################################

  covars <- colnames(train_data)

  ######################################################################################################
  #
  # Kluge to do the "Basic Genetic" & "Basic Binarized" covariate sets. It's "No Labs" & "Simple", with the following adjustment (comment out otherwise):
    chemo_pos <- grepl("CHEMO_", covars)
  
    train_data[,chemo_pos] <- ifelse(train_data[,chemo_pos] < 28, 1, 0)
    hasChemo <- rowSums(train_data[,chemo_pos])
    hasChemo <- ifelse(hasChemo > 0, 1, 0)
    train_data[,chemo_pos] <- list(NULL)
    train_data$hasChemo <- hasChemo
  
    val_data[,chemo_pos] <- ifelse(val_data[,chemo_pos] < 28, 1, 0)
    hasChemo <- rowSums(val_data[,chemo_pos])
    hasChemo <- ifelse(hasChemo > 0, 1, 0)
    val_data[,chemo_pos] <- list(NULL)
    val_data$hasChemo <- hasChemo
  
    train_data$DX_delta <- NULL
    val_data$DX_delta <- NULL
  
    covars <- colnames(train_data)
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

  
  # C-stat calculations
  num_samples <- length(out$outcomes_vte_test)
  sample_index <- seq(1, num_samples)

  out$c_index_vte <- list()
  out$c_index_vte2 <- list()
  out$c_index_death <- list()
  out$AUC <- list()
  out$int_risk_sens <- list()
  out$int_risk_spec <- list()
  out$high_risk_sens <- list()
  out$high_risk_spec <- list()

  for(i in seq(1,200))
  {
      test_seq <- sample(sample_index, num_samples, replace=T)
      preds_vte <- predictions_test$predicted_vte[test_seq,]
      obs <- data$OBS_TIME_6[test_seq]
      outcms_vte <- out$outcomes_vte_test[test_seq]

      event <- data$EVENT_6[test_seq]

      # Old Harrell c-index (in intsurv)
      if (algo == "rsf")
      {
         # rsf
         risk_scr = predictions_test$predicted$predicted[test_seq,1]
         out$c_index_vte[i] <- cIndex(time = obs, event = event, risk_score = risk_scr, weight = NULL)[1]   
      }else
      {    
         # fg
         risk_scr = predictions_test$predicted_vte[test_seq,1]
         out$c_index_vte[i] <- cIndex(time = obs, event = event, risk_score = risk_scr, weight = NULL)[1]   
      }

      # Antolini
      out$c_index_vte2[i] <- c_index(preds_vte, obs, outcms_vte)

      if (!is.null(dim(predictions_test$predicted_death)))
      {
         preds_death <- predictions_test$predicted_death[test_seq,]
         outcms_death <- out$outcomes_death_test[test_seq]

        if(sum(outcms_death) == 0)
        {
           # Fix a divide by zero scenario in concordance_td, in "pycox/pycox/evaluation/concordance.py"
           out$c_index_death[i] <- 0
        }else
        {
           out$c_index_death[i] <- c_index(preds_death, obs, outcms_death)
        }
      }else
      {
        out$c_index_death[i] <- NA
      }

      # Get the AUC
      if (algo == "rsf")
      {
         cif_index <- max(which(predictions_test$predicted$time.interest <= 180))
         cif1 <- predictions_test$predicted$cif[test_seq,cif_index,1]
      }else
      {
         # algo=FG
         cif1 <- predictions_test$predicted_vte[test_seq,180]
      }
      pred_vte_cif <- prediction(cif1, outcms_vte)
      out$AUC[i] <- unlist(performance(pred_vte_cif,"auc")@y.values)

      
      # Compute risk sensitivities
      # str(performance(pred_vte_cif,"auc")@y.values) # also see y.name
      rocCurve <- performance(pred_vte_cif,"tpr", "fpr")
      fpr <- unlist(rocCurve@x.values)   # false positive rate
      tpr <- unlist(rocCurve@y.values)   # true positive rate
      sens <- tpr
      spec <- 1 - fpr

      event_vte <- event
      event_vte[event_vte==2] = 0
      prevalence = sum(event_vte) / length(event_vte)

      ppv = sens * prevalence / (sens * prevalence + fpr * (1 - prevalence))
      ppv[is.na(ppv) == TRUE] = 0

      if(any(ppv >= 0.089))
      {
            int_risk_sens = round(max(sens[ppv>=0.089]), 2)
            int_risk_spec = round(min(spec[ppv>=0.089]), 2)
            if(any(ppv>=0.11))
            {
                high_risk_sens = round(max(sens[ppv>=0.11]), 2)
                high_risk_spec = round(min(spec[ppv>=0.11]), 2)
            }else
            {
                high_risk_sens = 0
                high_risk_spec = 0
            }
      }else
      {
            int_risk_sens = 0
            int_risk_spec = 0
            high_risk_sens = 0
            high_risk_spec = 0
      }

      out$int_risk_sens[i] <- int_risk_sens
      out$int_risk_spec[i] <- int_risk_spec
      out$high_risk_sens[i] <- high_risk_sens
      out$high_risk_spec[i] <- high_risk_spec
  }

  out
  
}


# Fits a batch of models and evaluates on the training set using k-fold cross-validation, returns  C-indexes
batcher_kf <- function(train, train_y, val, val_y, algo, dataDir, outPrefix, LSB_JOBINDEX, mtry, nodesize, ntrees)
{
  print("starting batcher...")
  set.seed(123)
    
  train_kf <- paste0(dataDir, train)
  train_kf_y <- paste0(dataDir, train_y)
  val_kf <- paste0(dataDir, val)
  val_kf_y <- paste0(dataDir, val_y)

  output_kf <- tester(train_kf, train_kf_y, val_kf, val_kf_y, algo, mtry, nodesize, ntrees)

  AUC <- unlist(output_kf$AUC)
  cindex_vte_test <- unlist(output_kf$c_index_vte)   # Harrel
  cindex_vte_test2 <- unlist(output_kf$c_index_vte2) # Antolini

  int_risk_sens <- unlist(output_kf$int_risk_sens )
  int_risk_spec <- unlist(output_kf$int_risk_spec )
  high_risk_sens <- unlist(output_kf$high_risk_sens )
  high_risk_spec <- unlist(output_kf$high_risk_spec )

  if (algo == "rsf") cindex_death_test <- unlist(output_kf$c_index_death) else cindex_death_test <- NA
      
  outfile <- paste0("Out/testing", ".csv")
  outdata <- data.frame(cindex_vte_test = cindex_vte_test, cindex_vte_test2 = cindex_vte_test2, cindex_death_test = cindex_death_test, 
                        AUC = AUC, int_risk_sens = int_risk_sens, int_risk_spec = int_risk_spec, high_risk_sens = high_risk_sens,
                        high_risk_spec = high_risk_spec)
  fwrite(outdata, file = outfile, append = FALSE)


  write.table(" ", outfile, append=TRUE, row.names=FALSE, col.names=FALSE, sep=" ", quote=FALSE)
  write.table(summary(outdata), outfile, append=TRUE, row.names=FALSE, sep=" ", quote=FALSE)
  write.table("\nStd Deviations:", outfile, append=TRUE, row.names=FALSE, col.names=FALSE, sep=" ", quote=FALSE)

  x <- t(as.data.frame(sqrt(diag(cov(outdata)))))
  write.table(x, outfile, append=TRUE, row.names=FALSE, sep=",", quote=FALSE)

  writeLines("\nFinished...")

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

