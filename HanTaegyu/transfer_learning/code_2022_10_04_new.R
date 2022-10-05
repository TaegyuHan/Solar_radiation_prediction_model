# --------------------------------------------- #
# 2022-10-03 : 순천향대학교 전이학습 모델
# 
# --------------------------------------------- #
rm(list = ls())

start_time <- Sys.time()

pc_number <- 2
DEFAULT_PATH <- "C:\\Users\\student\\Desktop\\solar\\Solar_radiation_prediction_model\\HanTaegyu\\transfer_learning"

# --------------------------------------------- #
# package
# install.packages("ranger")
# install.packages("rlist")
# install.packages("doParallel")
# install.packages('future.apply')
library(future)
library(tidyverse)
library(ranger)
library(rlist)
library(foreach) # Supporting the multithreading for XGBoost
library(doParallel) 
library(future.apply)
library(dplyr)
library(stringr)
# --------------------------------------------- #
# 질문
# 1. 교수님의 따릉이는 날짜데이터를 뺏는데, 현재 데이터도 날짜데이터를 빼도 될지?
# 2. csv로 빼는 부분에서 왜 시점 23과 24는 같은지? 이것이 의미하는 것이 무엇인지?
#

# --------------------------------------------- # 
# 데이터 불러오기

# 프로젝트 경로
setwd(DEFAULT_PATH)
getwd()
# [1] "D:/githubManagement/Solar_radiation_prediction_model/HanTaegyu/transfer_learning"

DATA_FOLDER_NAME <- "\\data"  # 데이터 폴더 이름
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
  ls(original.data)[idx] %>% print()   # 변수이름
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
        rbind(original.data[[idx]] %>% filter(Year >= 2019))
      next
    }
    
    train_data <- train_data %>%
      rbind(original.data[[idx]] %>% filter(Year < 2019))
    print(value.names[[idx]])
    print(original.data[[idx]] %>% filter(Year < 2019) %>% nrow())
  }
  
  print("예측해야 하는 도시의 크기")
  print(city_name)
  print(test_data %>% nrow())
  
  return(list(train = train_data, test = test_data))
}

###############################################################################
#                               서 울 예 측                                   #
###############################################################################

# Seoul을 제외한 데이터를 합함
result <- split_data("Daejeon")
train.data <- result$train
test.data <- result$test

# [1] Busan
# [1] 31255
# [1] Daegu
# [1] 32032
# [1] Daejeon
# [1] 32086
# [1] Gwangju
# [1] 32025
# [1] Incheon
# [1] 31987
# [1] "Seoul"
# [1] 8027

train.data %>% nrow()
test.data %>% nrow()

original.data$Busan %>% ncol()

loopnum <- nrow(test.data)
# [1] 7964

data.pred <- c(
  'data.pred01', 'data.pred02', 'data.pred03', 'data.pred04',
  'data.pred05', 'data.pred06', 'data.pred07', 'data.pred08',
  'data.pred09', 'data.pred10', 'data.pred11'
)

for(idx in (1:(data.pred %>% length()))){
  assign(data.pred[idx], numeric(loopnum))
}


# 데이터 나누기
multi_pc_data_index <- split(seq(loopnum), ceiling(seq(loopnum)/loopnum %/% 6))

for (i in 1:6) {
  paste0(
    multi_pc_data_index[[i]] %>% min() %>% as.character(),
    ":",
    multi_pc_data_index[[i]] %>% max() %>% as.character(),
    "     PC",
    i
  ) %>% print()
}
# [1] "1:1327     PC1"
# [1] "1328:2654     PC2"
# [1] "2655:3981     PC3"
# [1] "3982:5308     PC4"
# [1] "5309:6635     PC5"
# [1] "6636:(7964 - 11)     PC6"

if (pc_number == 1) {
  run_seq <- 1:1327  
} else if (pc_number == 2) {
  run_seq <- 1328:2654
} else if (pc_number == 3) {
  run_seq <- 2655:3981
} else if (pc_number == 4) {
  run_seq <- 3982:5308
} else if (pc_number == 5) {
  run_seq <- 5309:6635
} else if (pc_number == 6) {
  run_seq <- 6636:(7964 - 11)
}

# test.data %>% nrow()
# 1:7964


# result01 %>% nrow()
# 1:7964
# 1:7953
# [1] 7953

# result02 %>% nrow()
# 1:7964
# 2:7954
# [1] 7953

# result03 %>% nrow()
# 1:7964
# 3:7955
# [1] 7953

# result03 %>% nrow()
# 1:7964
# 4:7956
# [1] 7953

# result04 %>% nrow()
# 1:7964
# 5:7957
# [1] 7953

# result05 %>% nrow()
# 1:7964
# 6:7958
# [1] 7953

# result06 %>% nrow()
# 1:7964
# 7:7959
# [1] 7953

# result07 %>% nrow()
# 1:7964
# 8:7960
# [1] 7953

# result08 %>% nrow()
# 1:7964
# 9:7961
# [1] 7953

# result09 %>% nrow()
# 1:7964
# 10:7962
# [1] 7953

# result10 %>% nrow()
# 1:7964
# 11:7963
# [1] 7953

# result11 %>% nrow()
# 1:7964
# 12:7964
# [1] 7953

# 11시점을 예측하기 위해서 11 만큼만 봐줌
for(i in run_seq){
  i %>% print()
  model.ranger <- ranger(
    Solar~., data = rbind(train.data, rbind(train.data, test.data[i:(i + 10),])),
    num.tree = 128,
    mtry = 2,
    importance = "impurity"
  )

  # method 1
  # for(idx in 1:data.pred %>% length()){
  #   get(data.pred[idx])[i] <- predict(model.ranger, data = test.data[i+idx-1,])
  # }

  # method 2
  data.pred01[i] <- predict(model.ranger, data = test.data[i + 11,]) %>% as.numeric()
  data.pred02[i] <- predict(model.ranger, data = test.data[i + 12,]) %>% as.numeric()
  data.pred03[i] <- predict(model.ranger, data = test.data[i + 13,]) %>% as.numeric()
  data.pred04[i] <- predict(model.ranger, data = test.data[i + 14,]) %>% as.numeric()
  data.pred05[i] <- predict(model.ranger, data = test.data[i + 15,]) %>% as.numeric()
  data.pred06[i] <- predict(model.ranger, data = test.data[i + 16,]) %>% as.numeric()
  data.pred07[i] <- predict(model.ranger, data = test.data[i + 17,]) %>% as.numeric()
  data.pred08[i] <- predict(model.ranger, data = test.data[i + 18,]) %>% as.numeric()
  data.pred09[i] <- predict(model.ranger, data = test.data[i + 19,]) %>% as.numeric()
  data.pred10[i] <- predict(model.ranger, data = test.data[i + 20,]) %>% as.numeric()
  data.pred11[i] <- predict(model.ranger, data = test.data[i + 21,]) %>% as.numeric()
}

# 결과 저장

save(data.pred01, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "01", ".Rdata"))
save(data.pred02, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "02", ".Rdata"))
save(data.pred03, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "03", ".Rdata"))
save(data.pred04, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "04", ".Rdata"))
save(data.pred05, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "05", ".Rdata"))
save(data.pred06, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "06", ".Rdata"))
save(data.pred07, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "07", ".Rdata"))
save(data.pred08, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "08", ".Rdata"))
save(data.pred09, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "09", ".Rdata"))
save(data.pred10, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "10", ".Rdata"))
save(data.pred11, file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "11", ".Rdata"))

finish_time <- Sys.time() - start_time
finish_time %>% print()

# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "01", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "02", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "03", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "04", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "05", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "06", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "07", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "08", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "09", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "10", ".Rdata"))
# load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "11", ".Rdata"))

# 각각의 번호 별로 나눠서 저장하기

# result01 <- data.frame()
# result02 <- data.frame()
# result03 <- data.frame()
# result04 <- data.frame()
# result05 <- data.frame()
# result06 <- data.frame()
# result07 <- data.frame()
# result08 <- data.frame()
# result09 <- data.frame()
# result10 <- data.frame()
# result11 <- data.frame()
# 
# result.num <- c(
#   'result01', 'result02', 'result03', 'result04',
#   'result05', 'result06', 'result07', 'result08',
#   'result09', 'result10', 'result11'
# )
# 
# for(i in 1:6){
#   pc_number <- i
#   
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "01", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "02", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "03", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "04", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "05", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "06", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "07", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "08", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "09", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "10", ".Rdata"))
#   load(file = paste0(".\\predict", "\\", "PC", pc_number %>% as.character(), "_predict", "11", ".Rdata"))
#   
#   if (pc_number == 1) {
#     run_seq <- 1:1336  
#   } else if (pc_number == 2) {
#     run_seq <- 1337:2672  
#   } else if (pc_number == 3) {
#     run_seq <- 2673:4008
#   } else if (pc_number == 4) {
#     run_seq <- 4009:5344  
#   } else if (pc_number == 5) {
#     run_seq <- 5345:6680
#   } else if (pc_number == 6) {
#     run_seq <- 6681:8016
#   }
#   
#   # method1
#   # result01 <- rbind(result01, as.data.frame(data.pred01[run_seq]))
#   # result02 <- rbind(result02, as.data.frame(data.pred02[run_seq]))
#   # result03 <- rbind(result03, as.data.frame(data.pred03[run_seq]))
#   # result04 <- rbind(result04, as.data.frame(data.pred04[run_seq]))
#   # result05 <- rbind(result05, as.data.frame(data.pred05[run_seq]))
#   # result06 <- rbind(result06, as.data.frame(data.pred06[run_seq]))
#   # result07 <- rbind(result07, as.data.frame(data.pred07[run_seq]))
#   # result08 <- rbind(result08, as.data.frame(data.pred08[run_seq]))
#   # result09 <- rbind(result09, as.data.frame(data.pred09[run_seq]))
#   # result10 <- rbind(result10, as.data.frame(data.pred10[run_seq]))
#   # result11 <- rbind(result11, as.data.frame(data.pred11[run_seq]))
#   
#   
#   # method2 -> 반복문을 활용해서 위의 반복을 줄임
#   for(idx in 1:11){
#     assign(
#       result.num[idx],
#       rbind(get(result.num[idx]), as.data.frame(get(data.pred[idx])[run_seq]))
#     )
#   }
#   
# }
# names(result01) <- "Solar"
# names(result02) <- "Solar"
# names(result03) <- "Solar"
# names(result04) <- "Solar"
# names(result05) <- "Solar"
# names(result06) <- "Solar"
# names(result07) <- "Solar"
# names(result08) <- "Solar"
# names(result09) <- "Solar"
# names(result10) <- "Solar"
# names(result11) <- "Solar"
# 
# nameresult01 %>% nrow()
# nameresult02 %>% nrow()
# nameresult03 %>% nrow()
# nameresult04 %>% nrow()
# nameresult05 %>% nrow()
# nameresult06 %>% nrow()
# nameresult07 %>% nrow()
# nameresult08 %>% nrow()
# nameresult09 %>% nrow()
# nameresult10 %>% nrow()
# nameresult11 %>% nrow()
# # [1] 7953
# 
# 
# # 결과 저장
# 
# 
# for (i in 1:11) {
#   if (i < 10){
#     paste0(
#       "Actual <- as.numeric(test.data$Solar[",i,":",7953 + i,"])\n",
#       "Forecast <- as.numeric(result",
#       i,
#       "$Solar)\n",
#       "Error <- Actual - Forecast\n",
#       "R2_0",i," <- rsq(Actual, Forecast)\n",
#       "RMSE_0",i," <- sqrt(mean(Error^2))\n",
#       "MAE_0",i," <- mean(abs(Error))\n",
#       "NRMSE_0",i," <- (RMSE_",i,"/mean(Actual))*100\n",
#       "Results <- data.frame(Actual, Forecast, Error, R2_0",i,", RMSE_0",i,", MAE_0",i,", NRMSE_0",i,")\n",
#       'write.csv(Results, "0',
#       i,
#       "_result.csv",
#       '", row.names = FALSE)\n',
#       "\n"
#     ) %>% cat()    
#   } else {
#     paste0(
#       "Actual <- as.numeric(test.data$Solar[",i,":",7953 + i,"])\n",
#       "Forecast <- as.numeric(result",
#       i,
#       "$Solar)\n",
#       "Error <- Actual - Forecast\n",
#       "R2_",i," <- rsq(Actual, Forecast)\n",
#       "RMSE_",i," <- sqrt(mean(Error^2))\n",
#       "MAE_",i," <- mean(abs(Error))\n",
#       "NRMSE_",i," <- (RMSE_",i,"/mean(Actual))*100\n",
#       "Results <- data.frame(Actual, Forecast, Error, R2_",i,", RMSE_",i,", MAE_",i,", NRMSE_",i,")\n",
#       'write.csv(Results, "',
#       i,
#       "_result.csv",
#       '", row.names = FALSE)\n',
#       "\n"
#     ) %>% cat()    
#   }
# }
# 
# rsq <- function (x, y) cor(x, y) ^ 2
# 
# Actual <- as.numeric(test.data$Solar[1:7954])
# Forecast <- as.numeric(result1$Solar)
# Error <- Actual - Forecast
# R2_01 <- rsq(Actual, Forecast)
# RMSE_01 <- sqrt(mean(Error^2))
# MAE_01 <- mean(abs(Error))
# NRMSE_01 <- (RMSE_1/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_01, RMSE_01, MAE_01, NRMSE_01)
# write.csv(Results, "01_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[2:7955])
# Forecast <- as.numeric(result2$Solar)
# Error <- Actual - Forecast
# R2_02 <- rsq(Actual, Forecast)
# RMSE_02 <- sqrt(mean(Error^2))
# MAE_02 <- mean(abs(Error))
# NRMSE_02 <- (RMSE_2/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_02, RMSE_02, MAE_02, NRMSE_02)
# write.csv(Results, "02_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[3:7956])
# Forecast <- as.numeric(result3$Solar)
# Error <- Actual - Forecast
# R2_03 <- rsq(Actual, Forecast)
# RMSE_03 <- sqrt(mean(Error^2))
# MAE_03 <- mean(abs(Error))
# NRMSE_03 <- (RMSE_3/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_03, RMSE_03, MAE_03, NRMSE_03)
# write.csv(Results, "03_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[4:7957])
# Forecast <- as.numeric(result4$Solar)
# Error <- Actual - Forecast
# R2_04 <- rsq(Actual, Forecast)
# RMSE_04 <- sqrt(mean(Error^2))
# MAE_04 <- mean(abs(Error))
# NRMSE_04 <- (RMSE_4/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_04, RMSE_04, MAE_04, NRMSE_04)
# write.csv(Results, "04_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[5:7958])
# Forecast <- as.numeric(result5$Solar)
# Error <- Actual - Forecast
# R2_05 <- rsq(Actual, Forecast)
# RMSE_05 <- sqrt(mean(Error^2))
# MAE_05 <- mean(abs(Error))
# NRMSE_05 <- (RMSE_5/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_05, RMSE_05, MAE_05, NRMSE_05)
# write.csv(Results, "05_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[6:7959])
# Forecast <- as.numeric(result6$Solar)
# Error <- Actual - Forecast
# R2_06 <- rsq(Actual, Forecast)
# RMSE_06 <- sqrt(mean(Error^2))
# MAE_06 <- mean(abs(Error))
# NRMSE_06 <- (RMSE_6/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_06, RMSE_06, MAE_06, NRMSE_06)
# write.csv(Results, "06_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[7:7960])
# Forecast <- as.numeric(result7$Solar)
# Error <- Actual - Forecast
# R2_07 <- rsq(Actual, Forecast)
# RMSE_07 <- sqrt(mean(Error^2))
# MAE_07 <- mean(abs(Error))
# NRMSE_07 <- (RMSE_7/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_07, RMSE_07, MAE_07, NRMSE_07)
# write.csv(Results, "07_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[8:7961])
# Forecast <- as.numeric(result8$Solar)
# Error <- Actual - Forecast
# R2_08 <- rsq(Actual, Forecast)
# RMSE_08 <- sqrt(mean(Error^2))
# MAE_08 <- mean(abs(Error))
# NRMSE_08 <- (RMSE_8/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_08, RMSE_08, MAE_08, NRMSE_08)
# write.csv(Results, "08_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[9:7962])
# Forecast <- as.numeric(result9$Solar)
# Error <- Actual - Forecast
# R2_09 <- rsq(Actual, Forecast)
# RMSE_09 <- sqrt(mean(Error^2))
# MAE_09 <- mean(abs(Error))
# NRMSE_09 <- (RMSE_9/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_09, RMSE_09, MAE_09, NRMSE_09)
# write.csv(Results, "09_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[10:7963])
# Forecast <- as.numeric(result10$Solar)
# Error <- Actual - Forecast
# R2_10 <- rsq(Actual, Forecast)
# RMSE_10 <- sqrt(mean(Error^2))
# MAE_10 <- mean(abs(Error))
# NRMSE_10 <- (RMSE_10/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_10, RMSE_10, MAE_10, NRMSE_10)
# write.csv(Results, "10_result.csv", row.names = FALSE)
# 
# Actual <- as.numeric(test.data$Solar[11:7964])
# Forecast <- as.numeric(result11$Solar)
# Error <- Actual - Forecast
# R2_11 <- rsq(Actual, Forecast)
# RMSE_11 <- sqrt(mean(Error^2))
# MAE_11 <- mean(abs(Error))
# NRMSE_11 <- (RMSE_11/mean(Actual))*100
# Results <- data.frame(Actual, Forecast, Error, R2_11, RMSE_11, MAE_11, NRMSE_11)
# write.csv(Results, "11_result.csv", row.names = FALSE)
