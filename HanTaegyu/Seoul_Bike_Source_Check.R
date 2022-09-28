
# ---------------------------------------------------------------------------- #
# 변수 초기화
rm(list=ls())
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# 사용 라이브러리
library(tidyverse)
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
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
# 경로 설정하기
GITGUB_GIT_FOLDER_PATH <- "C:\\Users\\gksxo\\Desktop\\Folder\\github\\Solar_radiation_prediction_model\\HanTaegyu"
setwd(GITGUB_GIT_FOLDER_PATH)

# 랜덤변수 고정
set.seed(1234)


# 데이터 read 하기
dataset <- read.csv(".\\Bike_prepro.csv")


str(dataset)
# 'data.frame':	8760 obs. of  14 variables:
# $ Date                 : chr  "01/12/2017" "01/12/2017" "01/12/2017" "01/12/2017" ...
# $ Rented.Bike.Count    : int  254 204 173 107 78 100 181 460 930 490 ...
# $ Hour                 : int  0 1 2 3 4 5 6 7 8 9 ...
# $ Temperature          : num  -5.2 -5.5 -6 -6.2 -6 -6.4 -6.6 -7.4 -7.6 -6.5 ...
# $ Humidity             : int  37 38 39 40 36 37 35 38 37 27 ...
# $ Wind.speed           : num  2.2 0.8 1 0.9 2.3 1.5 1.3 0.9 1.1 0.5 ...
# $ Visibility           : int  2000 2000 2000 2000 2000 2000 2000 2000 2000 1928 ...
# $ Dew.point.temperature: num  -17.6 -17.6 -17.7 -17.6 -18.6 -18.7 -19.5 -19.3 -19.8 -22.4 ...
# $ Solar.Radiation      : num  0 0 0 0 0 0 0 0 0.01 0.23 ...
# $ Rainfall             : num  0 0 0 0 0 0 0 0 0 0 ...
# $ Snowfall             : num  0 0 0 0 0 0 0 0 0 0 ...
# $ Seasons              : int  4 4 4 4 4 4 4 4 4 4 ...
# $ Holiday              : int  1 1 1 1 1 1 1 1 1 1 ...
# $ Functioning.Day      : int  1 1 1 1 1 1 1 1 1 1 ...


# 데이터 프레임 size
dim(dataset)
# row : 8760
# col : 14   
df.row <- nrow(dataset)

# ---------------------------------- #
# Train, Test 나누기
train.size <- 2160
Training <- dataset[1:train.size,1:14]
Test <- dataset[train.size + 1:df.row,1:14]
# ----------------------------------

# ---------------------------------- #
# Randon Forests
# URL : https://www.rdocumentation.org/packages/ranger/versions/0.14.1/topics/ranger

start_time <- Sys.time()  # 모델 시간 체크
RF <- ranger::ranger(Rented.Bike.Count~.,
                     data = Training[2:14], 
                     num.trees = 128,  # Number of trees.
                     importance = "impurity")  # 가변 중요성 모드 "impurity"
end_time <- Sys.time()
Time_B1 <- end_time - start_time
capture.output(Time_B1, file = ".\\time_check\\SysTime_RF.txt", append = TRUE)  # 모델 시간 결과 저장하기
# Ranger result
# 
# Call:
#   ranger(Rented.Bike.Count ~ ., data = Training[2:14], num.trees = 128,      importance = "impurity") 
# 
# Type:                             Regression 
# Number of trees:                  128 
# Sample size:                      2160 
# Number of independent variables:  12 
# Mtry:                             3 
# Target node size:                 5 
# Variable importance mode:         impurity 
# Splitrule:                        variance 
# OOB prediction error (MSE):       7578.179 
# R squared (OOB):                  0.6648575 
# ---------------------------------- #

# ---------------------------------- #
# 예측 및 모델 결과 저장
Test %>% nrow()
# [1] 8760

# NA 값 제거
not.NA.test.row.index <- Test[2:14] %>% complete.cases()
# [1] 6600
# NA값 제거로 8760 > 6600 개로 변경
not.NA.Test <- Test[not.NA.test.row.index, 2:14]


# Traing NA 확인 
Training[Training[2:14] %>% complete.cases(), 2:14] %>% nrow()
# [1] 2160


# 예측
predict.result <- predict(RF, data = not.NA.Test)
# Ranger prediction
# 
# Type:                             Regression 
# Sample size:                      6600 
# Number of independent variables:  12


predict <- predict.result$predictions
# [1] 127.46229 178.41693 155.38627  85.81758  95.99645 ...

Actual <- as.data.frame(Test[not.NA.test.row.index,2]) # 실제 값
Prediction <- data.frame(predict) # 예측 값

str(Prediction)
# 'data.frame':	6600 obs. of  1 variable:
#   $ predict: num  127.5 178.4 155.4 85.8 96 ..

Date <- data.frame(Test[not.NA.test.row.index,1]) # 날짜
Hour <- data.frame(Test[not.NA.test.row.index,3]) # 시간

Result_RF_Ori_128 <- cbind(Date, Hour, Actual, Prediction)

str(Result_RF_Ori_128)
# 'data.frame':	6600 obs. of  4 variables:
#   $ Test.not.NA.test.row.index..1.: chr  "01/03/2018" "01/03/2018" "01/03/2018" "01/03/2018" ...
# $ Test.not.NA.test.row.index..3.: int  0 1 2 3 4 5 6 7 8 9 ...
# $ Test[not.NA.test.row.index, 2]: int  71 147 180 92 28 11 33 61 148 160 ...
# $ predict                       : num  127.5 178.4 155.4 85.8 96 ...
write.csv(Result_RF_Ori_128, ".\\model_result_csv\\ranger_unseen.csv", row.names = FALSE)
# ---------------------------------- #


# ---------------------------------- #
# XGBoost
Actual <- data.frame()
Prediction <- data.frame()
Date <- data.frame()
Hour <- data.frame()


# 모델 파라미터
param <-  expand.grid(nrounds = 100, 
                      max_depth = 5, 
                      eta = 0.2, 
                      gamma = 1000, 
                      colsample_bytree = 1, 
                      min_child_weight = 1, 
                      subsample = 1) # Set to the optimal values
# nrounds max_depth eta gamma colsample_bytree min_child_weight subsample
# 1     100         5 0.2  1000                1                1         1

start_time <- Sys.time()
# 병렬 클러스터 사용
# 자신의 PC 쓰레드에서 1개 적게 사용
cluster <- makeCluster(detectCores() - 1) # Number of cores, convention to leave 1 core for OS
registerDoParallel(cluster)

set.seed(1234)
XGB <- caret::train(Rented.Bike.Count~.,
                    data = Training[,2:14],
                    method = "xgbTree",
                    trControl = trainControl(method = "none"),
                    tuneGrid = param  # param 사용
)

stopCluster(cluster)
registerDoSEQ()
end_time <- Sys.time()
Time_B1 <- end_time - start_time
capture.output(Time_B1, file = ".\\time_check\\SysTime_XGBoost.txt", append = TRUE)

# ---------------------------------- #
# 예측
Prediction <- as.data.frame(predict(XGB, Test[not.NA.test.row.index,2:14]))
str(Prediction)
# 'data.frame':	6600 obs. of  1 variable:
#   $ predict(XGB, Test[not.NA.test.row.index, 2:14]): num  162.7 140.3 140 53.1 14.8 ...

Actual <- rbind(Actual, as.data.frame(Test[not.NA.test.row.index,2]))
Date <- rbind(Date, as.data.frame(Test[not.NA.test.row.index,1]))
Hour <- rbind(Hour, as.data.frame(Test[not.NA.test.row.index,3]))
Result_XGBoost <- cbind(Date, Hour, Actual, Prediction)
write.csv(Result_XGBoost, ".\\model_result_csv\\xgboost_unseen.csv", row.names = FALSE)
# ---------------------------------- #


## Cubist
Actual <- data.frame()
Date <- data.frame()
Hour <- data.frame()

# ---------------------------------- #
# 모델 실행
param <-  expand.grid(committees = 100, 
                      neighbors = 5) # Set to the optimal values

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
capture.output(Time_B1, file = ".\\time_check\\SysTime_Cubist.txt", append = TRUE)
# ---------------------------------- #

Prediction <- as.data.frame(predict(Cubist, Test[not.NA.test.row.index,2:14]))
Actual <- as.data.frame(Test[not.NA.test.row.index,2])
Date <- as.data.frame(Test[not.NA.test.row.index,1])
Hour <- as.data.frame(Test[not.NA.test.row.index,3])
Result_Cubist <- cbind(Date, Hour, Actual, Prediction)
write.csv(Result_Cubist, ".\\model_result_csv\\cubist_unseen.csv", row.names = FALSE)

library(vip)  # 가변 중요도 그림을 구성하기 위한 일반 프레임워크
library(pdp)  #  다양한 유형의 기계 학습 모델 시각화 라이브러리

p1_RF <- vip(RF, num_features = 12) + ggtitle("Random Forest")
p1_XGB <- vip(XGB, num_features = 12) + ggtitle("Extreme Gradient Boosting")
p1_Cubist <- vip(Cubist, num_features = 12) + ggtitle("Cubist")
a1 <- grid.arrange(p1_RF, p1_XGB, p1_Cubist, nrow = 1)

h1_RF <- partial(RF, pred.var = c("Hour", "Temperature"), plot = TRUE, chull = TRUE, plot.engine = "ggplot2") + ggtitle("Random Forest")
h1_XGB <- partial(XGB, pred.var = c("Hour", "Temperature"), plot = TRUE, chull = TRUE, plot.engine = "ggplot2") + ggtitle("Extreme Gradient Boosting")

# h1_cubist 시각화 시간 많이 걸림
h1_Cubist <- partial(Cubist, pred.var = c("Hour", "Temperature"), plot = TRUE, chull = TRUE, plot.engine = "ggplot2") + ggtitle("Cubist")
b1 <- grid.arrange(h1_RF, h1_XGB, h1_Cubist, nrow = 1)

# TableGrob (1 x 3) "arrange": 3 grobs
# z     cells    name           grob
# 1 1 (1-1,1-1) arrange gtable[layout]
# 2 2 (1-1,2-2) arrange gtable[layout]
# 3 3 (1-1,3-3) arrange gtable[layout]

## 2-Stage Model Construction
# version 2 요청
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
