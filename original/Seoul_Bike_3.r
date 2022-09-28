rm(list=ls())
library(ggplot2)
library(xgboost) # XGBoost, xgb
library(foreach) # Supporting the multithreading for XGBoost
library(doParallel) # Supporting the multithreading for XGBoost
library(rpart.plot)
library(RColorBrewer)
library(lime)
library(randomForest) # Random forest, rf
library(ranger)
library(pls) # Partial least squares, pls
library(rpart) # Recursive partitioning, rpart
library(ipred) # Bagged trees, treebag
library(gbm) # Boosted trees, gbm
library(earth) # MARS, earth
library(caret) # Bagged FDA, bagFDA (classification model)
library(Cubist)

setwd("C:\\Users\\johnn\\Dropbox\\문지훈\\Jihoon Moon\\CAU\\SCIE\\Cities\\")
dataset <- read.csv("C:\\Users\\johnn\\Dropbox\\문지훈\\Jihoon Moon\\CAU\\SCIE\\Cities\\Bike_prepro.csv")

Training <- dataset[1:2160,1:14]
Test <- dataset[2161:8760,1:14]

# Randon Forests
start_time <- Sys.time()
set.seed(1234)
RF <- ranger(Rented.Bike.Count~., data = Training[2:14], num.trees = 128, importance = "impurity")
end_time <- Sys.time()
Time_B1 <- end_time - start_time
capture.output(Time_B1, file = "SysTime_RF.txt", append = TRUE)

predict <- predict(RF,Test[2:14])
RF_predict <- predict$predictions

Actual <- as.data.frame(Test[,2])
Prediction <- data.frame(RF_predict)
Date <- data.frame(Test[,1])
Hour <- data.frame(Test[,3])
Result_RF_Ori_128 <- cbind(Date, Hour, Actual, Prediction)
write.csv(Result_RF_Ori_128, "ranger_unseen.csv", row.names = FALSE)

# XGBoost
Actual <- data.frame()
Prediction <- data.frame()
Date <- data.frame()
Hour <- data.frame()

param <-  expand.grid(nrounds = 100, max_depth = 5, eta = 0.2, gamma = 1000, 
                      colsample_bytree = 1, min_child_weight = 1, subsample = 1) # Set to the optimal values

start_time <- Sys.time()
cluster <- makeCluster(detectCores() - 1) # Number of cores, convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(1234)
XGB <- caret::train(Rented.Bike.Count~.,
                    data = Training[,2:14],
                    method = "xgbTree",
                    trControl = trainControl(method = "none"),
                    tuneGrid = param
)

stopCluster(cluster)
registerDoSEQ()
end_time <- Sys.time()
Time_B1 <- end_time - start_time
capture.output(Time_B1, file = "SysTime_XGBoost.txt", append = TRUE)

temp <- as.data.frame(predict(XGB, Test[,2:14]))
Prediction <- rbind(Prediction, temp)
Actual <- rbind(Actual, as.data.frame(Test[,2]))
Date <- rbind(Date, as.data.frame(Test[,1]))
Hour <- rbind(Hour, as.data.frame(Test[,3]))
Result_XGBoost <- cbind(Date, Hour, Actual, Prediction)
write.csv(Result_XGBoost, "xgboost_unseen.csv", row.names = FALSE)

## Cubist
Actual <- data.frame()
Prediction <- data.frame()
Date <- data.frame()
Hour <- data.frame()

param <-  expand.grid(committees = 100, neighbors = 5) # Set to the optimal values

start_time <- Sys.time()
cluster <- makeCluster(detectCores() - 1) # Number of cores, convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(1234)
Cubist <- caret::train(Rented.Bike.Count~.,
                    data = Training[,2:14],
                    method = "cubist",
                    trControl = trainControl(method = "none"),
                    tuneGrid = param
)
stopCluster(cluster)
registerDoSEQ()

end_time <- Sys.time()
Time_B1 <- end_time - start_time
capture.output(Time_B1, file = "SysTime_Cubist.txt", append = TRUE)

temp <- as.data.frame(predict(Cubist, Test[,2:14]))
Prediction <- rbind(Prediction, temp)
Actual <- rbind(Actual, as.data.frame(Test[,2]))
Date <- rbind(Date, as.data.frame(Test[,1]))
Hour <- rbind(Hour, as.data.frame(Test[,3]))
Result_Cubist <- cbind(Date, Hour, Actual, Prediction)
write.csv(Result_Cubist, "cubist_unseen.csv", row.names = FALSE)

library(vip)
library(pdp)
p1_RF <- vip(RF, num_features = 12) + ggtitle("Random Forest")
p1_XGB <- vip(XGB, num_features = 12) + ggtitle("Extreme Gradient Boosting")
p1_Cubist <- vip(Cubist, num_features = 12) + ggtitle("Cubist")
a1 <- grid.arrange(p1_RF, p1_XGB, p1_Cubist, nrow = 1)

h1_RF <- partial(RF, pred.var = c("Hour", "Temperature"), plot = TRUE, chull = TRUE, plot.engine = "ggplot2") + ggtitle("Random Forest")
h1_XGB <- partial(XGB, pred.var = c("Hour", "Temperature"), plot = TRUE, chull = TRUE, plot.engine = "ggplot2") + ggtitle("Extreme Gradient Boosting")
h1_Cubist <- partial(Cubist, pred.var = c("Hour", "Temperature"), plot = TRUE, chull = TRUE, plot.engine = "ggplot2") + ggtitle("Cubist")
b1 <- grid.arrange(h1_RF, h1_XGB, h1_Cubist, nrow = 1)

## 2-Stage Model Construction
setwd("C:\\Users\\johnn\\Dropbox\\문지훈\\Jihoon Moon\\CAU\\SCIE\\Cities\\")
Data <- read.csv("C:\\Users\\Johnn\\Dropbox\\문지훈\\Jihoon Moon\\CAU\\SCIE\\Cities\\Bike_prepro_v2.csv")

Dataset <- Data[,c(2:17)]
ind_train <- round(2160)
loopnum <- nrow(Dataset)-ind_train
dataPred01 <- matrix(ncol=ncol(Dataset))
dataPred02 <- matrix(ncol=ncol(Dataset))
dataPred03 <- matrix(ncol=ncol(Dataset))
dataPred04 <- matrix(ncol=ncol(Dataset))
dataPred05 <- matrix(ncol=ncol(Dataset))
dataPred06 <- matrix(ncol=ncol(Dataset))
dataPred07 <- matrix(ncol=ncol(Dataset))
dataPred08 <- matrix(ncol=ncol(Dataset))
dataPred09 <- matrix(ncol=ncol(Dataset))
dataPred10 <- matrix(ncol=ncol(Dataset))
dataPred11 <- matrix(ncol=ncol(Dataset))
dataPred12 <- matrix(ncol=ncol(Dataset))
dataPred13 <- matrix(ncol=ncol(Dataset))
dataPred14 <- matrix(ncol=ncol(Dataset))
dataPred15 <- matrix(ncol=ncol(Dataset))
dataPred16 <- matrix(ncol=ncol(Dataset))
dataPred17 <- matrix(ncol=ncol(Dataset))
dataPred18 <- matrix(ncol=ncol(Dataset))
dataPred19 <- matrix(ncol=ncol(Dataset))
dataPred20 <- matrix(ncol=ncol(Dataset))
dataPred21 <- matrix(ncol=ncol(Dataset))
dataPred22 <- matrix(ncol=ncol(Dataset))
dataPred23 <- matrix(ncol=ncol(Dataset))
dataPred24 <- matrix(ncol=ncol(Dataset))

x1 <- matrix(ncol=ncol(Dataset))
trainData <- na.omit(Dataset[(1:ind_train),])
testData <- na.omit(Dataset[ind_train+1:nrow(Dataset),])

for(i in 1:loopnum)
{
  set.seed(1234)
#  ranger_tscv <- ranger(Rented.Bike.Count~., data = Dataset[1:(ind_train-1+i),], num.trees = 128, mtry = 1, importance = "impurity")
  ranger_tscv <- ranger(Rented.Bike.Count~., data = Dataset[(ind_train-168+i):(ind_train-1+i),], num.trees = 128, mtry = 5, importance = "impurity")
  dataPred01[i]<-predict(ranger_tscv, data = testData[i,])
  dataPred02[i]<-predict(ranger_tscv, data = testData[i+1,])
  dataPred03[i]<-predict(ranger_tscv, data = testData[i+2,])
  dataPred04[i]<-predict(ranger_tscv, data = testData[i+3,])
  dataPred05[i]<-predict(ranger_tscv, data = testData[i+4,])
  dataPred06[i]<-predict(ranger_tscv, data = testData[i+5,])
  dataPred07[i]<-predict(ranger_tscv, data = testData[i+6,])
  dataPred08[i]<-predict(ranger_tscv, data = testData[i+7,])
  dataPred09[i]<-predict(ranger_tscv, data = testData[i+8,])
  dataPred10[i]<-predict(ranger_tscv, data = testData[i+9,])
  dataPred11[i]<-predict(ranger_tscv, data = testData[i+10,])
  dataPred12[i]<-predict(ranger_tscv, data = testData[i+11,])
  dataPred13[i]<-predict(ranger_tscv, data = testData[i+12,])
  dataPred14[i]<-predict(ranger_tscv, data = testData[i+13,])
  dataPred15[i]<-predict(ranger_tscv, data = testData[i+14,])
  dataPred16[i]<-predict(ranger_tscv, data = testData[i+15,])
  dataPred17[i]<-predict(ranger_tscv, data = testData[i+16,])
  dataPred18[i]<-predict(ranger_tscv, data = testData[i+17,])
  dataPred19[i]<-predict(ranger_tscv, data = testData[i+18,])
  dataPred20[i]<-predict(ranger_tscv, data = testData[i+19,])
  dataPred21[i]<-predict(ranger_tscv, data = testData[i+20,])
  dataPred22[i]<-predict(ranger_tscv, data = testData[i+21,])
  dataPred23[i]<-predict(ranger_tscv, data = testData[i+22,])
  dataPred24[i]<-predict(ranger_tscv, data = testData[i+23,])
}
rsq <- function (x, y) cor(x, y) ^ 2

Actual <- as.numeric(testData$Rented.Bike.Count[1:6578])
Forecast <- as.numeric(dataPred01)
Error <- Actual - Forecast
R2_01 <- rsq(Actual, Forecast)
RMSE_01 <- sqrt(mean(Error^2))
MAE_01 <- mean(abs(Error))
NRMSE_01 <- (RMSE_01/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_01, RMSE_01, MAE_01, NRMSE_01)
write.csv(Results, "2-stage_ours_01.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[2:6579])
Forecast <- as.numeric(dataPred02)
Error <- Actual - Forecast
R2_02 <- rsq(Actual, Forecast)
RMSE_02 <- sqrt(mean(Error^2))
MAE_02 <- mean(abs(Error))
NRMSE_02 <- (RMSE_02/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_02, RMSE_02, MAE_02, NRMSE_02)
write.csv(Results, "2-stage_ours_02.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[3:6580])
Forecast <- as.numeric(dataPred03)
Error <- Actual - Forecast
R2_03 <- rsq(Actual, Forecast)
RMSE_03 <- sqrt(mean(Error^2))
MAE_03 <- mean(abs(Error))
NRMSE_03 <- (RMSE_03/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_03, RMSE_03, MAE_03, NRMSE_03)
write.csv(Results, "2-stage_ours_03.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[4:6581])
Forecast <- as.numeric(dataPred04)
Error <- Actual - Forecast
R2_04 <- rsq(Actual, Forecast)
RMSE_04 <- sqrt(mean(Error^2))
MAE_04 <- mean(abs(Error))
NRMSE_04 <- (RMSE_04/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_04, RMSE_04, MAE_04, NRMSE_04)
write.csv(Results, "2-stage_ours_04.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[5:6582])
Forecast <- as.numeric(dataPred05)
Error <- Actual - Forecast
R2_05 <- rsq(Actual, Forecast)
RMSE_05 <- sqrt(mean(Error^2))
MAE_05 <- mean(abs(Error))
NRMSE_05 <- (RMSE_05/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_05, RMSE_05, MAE_05, NRMSE_05)
write.csv(Results, "2-stage_ours_05.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[6:6583])
Forecast <- as.numeric(dataPred06)
Error <- Actual - Forecast
R2_06 <- rsq(Actual, Forecast)
RMSE_06 <- sqrt(mean(Error^2))
MAE_06 <- mean(abs(Error))
NRMSE_06 <- (RMSE_06/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_06, RMSE_06, MAE_06, NRMSE_06)
write.csv(Results, "2-stage_ours_06.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[7:6584])
Forecast <- as.numeric(dataPred07)
Error <- Actual - Forecast
R2_07 <- rsq(Actual, Forecast)
RMSE_07 <- sqrt(mean(Error^2))
MAE_07 <- mean(abs(Error))
NRMSE_07 <- (RMSE_07/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_07, RMSE_07, MAE_07, NRMSE_07)
write.csv(Results, "2-stage_ours_07.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[8:6585])
Forecast <- as.numeric(dataPred08)
Error <- Actual - Forecast
R2_08 <- rsq(Actual, Forecast)
RMSE_08 <- sqrt(mean(Error^2))
MAE_08 <- mean(abs(Error))
NRMSE_08 <- (RMSE_08/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_08, RMSE_08, MAE_08, NRMSE_08)
write.csv(Results, "2-stage_ours_08.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[9:6586])
Forecast <- as.numeric(dataPred09)
Error <- Actual - Forecast
R2_09 <- rsq(Actual, Forecast)
RMSE_09 <- sqrt(mean(Error^2))
MAE_09 <- mean(abs(Error))
NRMSE_09 <- (RMSE_09/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_09, RMSE_09, MAE_09, NRMSE_09)
write.csv(Results, "2-stage_ours_09.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[10:6587])
Forecast <- as.numeric(dataPred10)
Error <- Actual - Forecast
R2_10 <- rsq(Actual, Forecast)
RMSE_10 <- sqrt(mean(Error^2))
MAE_10 <- mean(abs(Error))
NRMSE_10 <- (RMSE_10/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_10, RMSE_10, MAE_10, NRMSE_10)
write.csv(Results, "2-stage_ours_10.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[11:6588])
Forecast <- as.numeric(dataPred11)
Error <- Actual - Forecast
R2_11 <- rsq(Actual, Forecast)
RMSE_11 <- sqrt(mean(Error^2))
MAE_11 <- mean(abs(Error))
NRMSE_11 <- (RMSE_11/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_11, RMSE_11, MAE_11, NRMSE_11)
write.csv(Results, "2-stage_ours_11.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[12:6589])
Forecast <- as.numeric(dataPred12)
Error <- Actual - Forecast
R2_12 <- rsq(Actual, Forecast)
RMSE_12 <- sqrt(mean(Error^2))
MAE_12 <- mean(abs(Error))
NRMSE_12 <- (RMSE_12/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_12, RMSE_12, MAE_12, NRMSE_12)
write.csv(Results, "2-stage_ours_12.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[13:6590])
Forecast <- as.numeric(dataPred13)
Error <- Actual - Forecast
R2_13 <- rsq(Actual, Forecast)
RMSE_13 <- sqrt(mean(Error^2))
MAE_13 <- mean(abs(Error))
NRMSE_13 <- (RMSE_13/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_13, RMSE_13, MAE_13, NRMSE_13)
write.csv(Results, "2-stage_ours_13.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[14:6591])
Forecast <- as.numeric(dataPred14)
Error <- Actual - Forecast
R2_14 <- rsq(Actual, Forecast)
RMSE_14 <- sqrt(mean(Error^2))
MAE_14 <- mean(abs(Error))
NRMSE_14 <- (RMSE_14/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_14, RMSE_14, MAE_14, NRMSE_14)
write.csv(Results, "2-stage_ours_14.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[15:6592])
Forecast <- as.numeric(dataPred15)
Error <- Actual - Forecast
R2_15 <- rsq(Actual, Forecast)
RMSE_15 <- sqrt(mean(Error^2))
MAE_15 <- mean(abs(Error))
NRMSE_15 <- (RMSE_15/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_15, RMSE_15, MAE_15, NRMSE_15)
write.csv(Results, "2-stage_ours_15.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[16:6593])
Forecast <- as.numeric(dataPred16)
Error <- Actual - Forecast
R2_16 <- rsq(Actual, Forecast)
RMSE_16 <- sqrt(mean(Error^2))
MAE_16 <- mean(abs(Error))
NRMSE_16 <- (RMSE_16/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_16, RMSE_16, MAE_16, NRMSE_16)
write.csv(Results, "2-stage_ours_16.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[17:6594])
Forecast <- as.numeric(dataPred17)
Error <- Actual - Forecast
R2_17 <- rsq(Actual, Forecast)
RMSE_17 <- sqrt(mean(Error^2))
MAE_17 <- mean(abs(Error))
NRMSE_17 <- (RMSE_17/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_17, RMSE_17, MAE_17, NRMSE_17)
write.csv(Results, "2-stage_ours_17.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[18:6595])
Forecast <- as.numeric(dataPred18)
Error <- Actual - Forecast
R2_18 <- rsq(Actual, Forecast)
RMSE_18 <- sqrt(mean(Error^2))
MAE_18 <- mean(abs(Error))
NRMSE_18 <- (RMSE_18/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_18, RMSE_18, MAE_18, NRMSE_18)
write.csv(Results, "2-stage_ours_18.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[19:6596])
Forecast <- as.numeric(dataPred19)
Error <- Actual - Forecast
R2_19 <- rsq(Actual, Forecast)
RMSE_19 <- sqrt(mean(Error^2))
MAE_19 <- mean(abs(Error))
NRMSE_19 <- (RMSE_19/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_19, RMSE_19, MAE_19, NRMSE_19)
write.csv(Results, "2-stage_ours_19.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[20:6597])
Forecast <- as.numeric(dataPred20)
Error <- Actual - Forecast
R2_20 <- rsq(Actual, Forecast)
RMSE_20 <- sqrt(mean(Error^2))
MAE_20 <- mean(abs(Error))
NRMSE_20 <- (RMSE_20/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_20, RMSE_20, MAE_20, NRMSE_20)
write.csv(Results, "2-stage_ours_20.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[21:6598])
Forecast <- as.numeric(dataPred21)
Error <- Actual - Forecast
R2_21 <- rsq(Actual, Forecast)
RMSE_21 <- sqrt(mean(Error^2))
MAE_21 <- mean(abs(Error))
NRMSE_21 <- (RMSE_21/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_21, RMSE_21, MAE_21, NRMSE_21)
write.csv(Results, "2-stage_ours_21.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[22:6599])
Forecast <- as.numeric(dataPred22)
Error <- Actual - Forecast
R2_22 <- rsq(Actual, Forecast)
RMSE_22 <- sqrt(mean(Error^2))
MAE_22 <- mean(abs(Error))
NRMSE_22 <- (RMSE_22/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_22, RMSE_22, MAE_22, NRMSE_22)
write.csv(Results, "2-stage_ours_22.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[23:6600])
Forecast <- as.numeric(dataPred23)
Error <- Actual - Forecast
R2_23 <- rsq(Actual, Forecast)
RMSE_23 <- sqrt(mean(Error^2))
MAE_23 <- mean(abs(Error))
NRMSE_23 <- (RMSE_23/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_23, RMSE_23, MAE_23, NRMSE_23)
write.csv(Results, "2-stage_ours_23.csv", row.names = FALSE)

Actual <- as.numeric(testData$Rented.Bike.Count[24:6600])
Forecast <- as.numeric(dataPred24)
Error <- Actual - Forecast
R2_24 <- rsq(Actual, Forecast)
RMSE_24 <- sqrt(mean(Error^2))
MAE_24 <- mean(abs(Error))
NRMSE_24 <- (RMSE_24/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_24, RMSE_24, MAE_24, NRMSE_24)
write.csv(Results, "2-stage_ours_24.csv", row.names = FALSE)

library(ggplot2)
setwd("C:\\Users\\johnn\\Dropbox\\문지훈\\Jihoon Moon\\CAU\\SCIE\\Cities\\")
dataset <- read.csv("C:\\Users\\johnn\\Dropbox\\문지훈\\Jihoon Moon\\CAU\\SCIE\\Cities\\CVRMSE.csv")
ggplot(dataset, aes(Points, Value))+geom_line(aes(color = Models))
