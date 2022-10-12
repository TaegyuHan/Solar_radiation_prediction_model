rm(list = ls())
library(future)
library(tidyverse)
library(ranger)
library(rlist)
library(foreach) # Supporting the multithreading for XGBoost
library(doParallel) 
library(future.apply)
library(dplyr)
library(stringr)

DEFAULT_PATH <- "C:\\Users\\user\\Dropbox\\문지훈\\Jihoon Moon\\SCHU\\캡스톤디자인"
setwd(DEFAULT_PATH)
getwd()

DATA_FOLDER_NAME <- "\\Datasets"  # 데이터 폴더 이름
DATA_PATH <- paste0(DEFAULT_PATH, DATA_FOLDER_NAME) # 데이터 폴더 경로
files <-list.files(DATA_PATH) # 데이터 파일 이름들

# 파일 경로들
FILES_PATH <- paste0(DATA_PATH, paste0("\\", files))


# 데이터 읽기
original.data <- list()
for(idx in FILES_PATH %>% length() %>% seq()) {      
  original.data[[idx]] <- read.csv(file = FILES_PATH[idx],
                                   header = TRUE)
}

value.names <- str_remove(files, ".csv")
names(original.data) <- value.names

# 데이터 크기 확인
for(idx in FILES_PATH %>% length() %>% seq()) {
  ls(original.data)[idx] %>% print()   # 변수명
  original.data[[idx]] %>% dim() %>% print()  # 데이터 프레임 크기
}
# [1] "Busan"
# [1] 39284     9
# [1] "Daegu"
# [1] 40040     9
# [1] "Daejeon"
# [1] 40050     9
# [1] "Gwangju"
# [1] 40040     9
# [1] "Incheon"
# [1] 40025     9
# [1] "Seoul"
# [1] 39937     9

original.data$Busan %>% names()
# [1] "Year"  "Month" "Day"   "Hour"  "Temp"  "Humi"  "WS"    "WD"    "Solar"

# 데이터 시간순으로 정렬
for(idx in FILES_PATH %>% length() %>% seq()) {
  original.data[[idx]] <- original.data[[idx]] %>% 
    arrange("Year", "Month", "Day", "Hour")
}

# 데이터 추출
split_data <- function(city_name){
  train_data <- data.frame()
  test_data <- data.frame()
  
  print("각각의 도시의 크기 - 예측 도시 제외")
  
  for(idx in FILES_PATH %>% length() %>% seq()) {
    if (value.names[idx] == city_name) {
      test_data <- test_data %>%
        rbind(original.data[[idx]] %>% filter(Year >= 2020))
      next
    }
    
    train_data <- train_data %>%
      rbind(original.data[[idx]] %>% filter(Year < 2020))
    print(value.names[[idx]])
    print(original.data[[idx]] %>% filter(Year < 2020) %>% nrow())
  }
  
  print("예측해야 하는 도시의 크기")
  print(city_name)
  print(test_data %>% nrow())
  
  return(list(train = train_data, test = test_data))
}

# Daejeon을 제외한 데이터를 합함
result <- split_data("Daejeon")
train.data <- result$train
test.data <- result$test

write.csv(train.data, "Daejeon_training.csv", row.names = FALSE)
write.csv(test.data, "Daejeon_test.csv", row.names = FALSE)

ind_train <- round(11)
loopnum <- nrow(test.data)
dataPred01 <- matrix(ncol=ncol(test.data))
dataPred02 <- matrix(ncol=ncol(test.data))
dataPred03 <- matrix(ncol=ncol(test.data))
dataPred04 <- matrix(ncol=ncol(test.data))
dataPred05 <- matrix(ncol=ncol(test.data))
dataPred06 <- matrix(ncol=ncol(test.data))
dataPred07 <- matrix(ncol=ncol(test.data))
dataPred08 <- matrix(ncol=ncol(test.data))
dataPred09 <- matrix(ncol=ncol(test.data))
dataPred10 <- matrix(ncol=ncol(test.data))
dataPred11 <- matrix(ncol=ncol(test.data))

trainData <- train.data[,2:9]
testData <- test.data[,2:9]

# Ranger Model Construction
for(i in 1:loopnum)
{
  set.seed(1234)
  ranger_tscv <- ranger(Solar~., data = rbind(testData[i:(i+10),]), num.trees = 128, importance = "impurity")
  dataPred01[i]<-predict(ranger_tscv, data = testData[i+11,])
  dataPred02[i]<-predict(ranger_tscv, data = testData[i+12,])
  dataPred03[i]<-predict(ranger_tscv, data = testData[i+13,])
  dataPred04[i]<-predict(ranger_tscv, data = testData[i+14,])
  dataPred05[i]<-predict(ranger_tscv, data = testData[i+15,])
  dataPred06[i]<-predict(ranger_tscv, data = testData[i+16,])
  dataPred07[i]<-predict(ranger_tscv, data = testData[i+17,])
  dataPred08[i]<-predict(ranger_tscv, data = testData[i+18,])
  dataPred09[i]<-predict(ranger_tscv, data = testData[i+19,])
  dataPred10[i]<-predict(ranger_tscv, data = testData[i+20,])
  dataPred11[i]<-predict(ranger_tscv, data = testData[i+21,])
}

rsq <- function (x, y) cor(x, y) ^ 2

Actual <- as.numeric(testData$Solar[12:3951])
Forecast <- as.numeric(dataPred01)
Error <- Actual - Forecast
R2_01 <- rsq(Actual, Forecast)
RMSE_01 <- sqrt(mean(Error^2))
MAE_01 <- mean(abs(Error))
NRMSE_01 <- (RMSE_01/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_01, RMSE_01, MAE_01, NRMSE_01)
write.csv(Results, "handab_01.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[13:3952]))
Forecast <- na.omit(as.numeric(dataPred02))
Error <- Actual - Forecast
R2_02 <- rsq(Actual, Forecast)
RMSE_02 <- sqrt(mean(Error^2))
MAE_02 <- mean(abs(Error))
NRMSE_02 <- (RMSE_02/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_02, RMSE_02, MAE_02, NRMSE_02)
write.csv(Results, "handab_02.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[14:3953]))
Forecast <- na.omit(as.numeric(dataPred03))
Error <- Actual - Forecast
R2_03 <- rsq(Actual, Forecast)
RMSE_03 <- sqrt(mean(Error^2))
MAE_03 <- mean(abs(Error))
NRMSE_03 <- (RMSE_03/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_03, RMSE_03, MAE_03, NRMSE_03)
write.csv(Results, "handab_03.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[15:3954]))
Forecast <- na.omit(as.numeric(dataPred04))
Error <- Actual - Forecast
R2_04 <- rsq(Actual, Forecast)
RMSE_04 <- sqrt(mean(Error^2))
MAE_04 <- mean(abs(Error))
NRMSE_04 <- (RMSE_04/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_04, RMSE_04, MAE_04, NRMSE_04)
write.csv(Results, "handab_04.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[16:3955]))
Forecast <- na.omit(as.numeric(dataPred05))
Error <- Actual - Forecast
R2_05 <- rsq(Actual, Forecast)
RMSE_05 <- sqrt(mean(Error^2))
MAE_05 <- mean(abs(Error))
NRMSE_05 <- (RMSE_05/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_05, RMSE_05, MAE_05, NRMSE_05)
write.csv(Results, "handab_05.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[17:3956]))
Forecast <- na.omit(as.numeric(dataPred06))
Error <- Actual - Forecast
R2_06 <- rsq(Actual, Forecast)
RMSE_06 <- sqrt(mean(Error^2))
MAE_06 <- mean(abs(Error))
NRMSE_06 <- (RMSE_06/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_06, RMSE_06, MAE_06, NRMSE_06)
write.csv(Results, "handab_06.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[18:3957]))
Forecast <- na.omit(as.numeric(dataPred07))
Error <- Actual - Forecast
R2_07 <- rsq(Actual, Forecast)
RMSE_07 <- sqrt(mean(Error^2))
MAE_07 <- mean(abs(Error))
NRMSE_07 <- (RMSE_07/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_07, RMSE_07, MAE_07, NRMSE_07)
write.csv(Results, "handab_07.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[19:3958]))
Forecast <- na.omit(as.numeric(dataPred08))
Error <- Actual - Forecast
R2_08 <- rsq(Actual, Forecast)
RMSE_08 <- sqrt(mean(Error^2))
MAE_08 <- mean(abs(Error))
NRMSE_08 <- (RMSE_08/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_08, RMSE_08, MAE_08, NRMSE_08)
write.csv(Results, "handab_08.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[20:3959]))
Forecast <- na.omit(as.numeric(dataPred09))
Error <- Actual - Forecast
R2_09 <- rsq(Actual, Forecast)
RMSE_09 <- sqrt(mean(Error^2))
MAE_09 <- mean(abs(Error))
NRMSE_09 <- (RMSE_09/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_09, RMSE_09, MAE_09, NRMSE_09)
write.csv(Results, "handab_09.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[21:3960]))
Forecast <- na.omit(as.numeric(dataPred10))
Error <- Actual - Forecast
R2_10 <- rsq(Actual, Forecast)
RMSE_10 <- sqrt(mean(Error^2))
MAE_10 <- mean(abs(Error))
NRMSE_10 <- (RMSE_10/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_10, RMSE_10, MAE_10, NRMSE_10)
write.csv(Results, "handab_10.csv", row.names = FALSE)

Actual <- na.omit(as.numeric(testData$Solar[22:3961]))
Forecast <- na.omit(as.numeric(dataPred11))
Error <- Actual - Forecast
R2_11 <- rsq(Actual, Forecast)
RMSE_11 <- sqrt(mean(Error^2))
MAE_11 <- mean(abs(Error))
NRMSE_11 <- (RMSE_11/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_11, RMSE_11, MAE_11, NRMSE_11)
write.csv(Results, "handab_11.csv", row.names = FALSE)

testData <- test.data[,2:9]
testData_01 <- test.data[12:3960, 2:9]

Actual <- na.omit(as.numeric(testData_01$Solar))
Forecast <- na.omit(as.numeric(testData[1:3949,8]))
Error <- Actual - Forecast
R2_Per <- rsq(Actual, Forecast)
RMSE_Per <- sqrt(mean(Error^2))
MAE_Per <- mean(abs(Error))
NRMSE_Per <- (RMSE_Per/mean(Actual))*100
Results <- data.frame(Actual, Forecast, Error, R2_Per, RMSE_Per, MAE_Per, NRMSE_Per)
write.csv(Results, "handab_Per.csv", row.names = FALSE)