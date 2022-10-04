# --------------------------------------------- #
# 2022-10-03 : 순천향대학교 전이학습 모델
# 
# --------------------------------------------- #
rm(list = ls())
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
#
#

# --------------------------------------------- # 
# 데이터 불러오기

# 프로젝트 경로
DEFAULT_PATH <- "D:\\githubManagement\\Solar_radiation_prediction_model\\HanTaegyu\\transfer_learning"
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
result <- split_data("Seoul")
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

input_Value_cnt <- original.data$Busan %>% names() %>% length()
loopnum <- nrow(test.data)

data.pred <- c(
  'data.pred01', 'data.pred02', 'data.pred03', 'data.pred04',
  'data.pred05', 'data.pred06', 'data.pred07', 'data.pred08',
  'data.pred09', 'data.pred10', 'data.pred11'
)

for(idx in 1:data.pred %>% length()){
  assign(data.pred[idx], matrix(ncol = input_Value_cnt))
}

# 11시점을 예측하기 위해서 11 만큼만 봐줌
for(i in 1:loopnum){
  model.ranger <- ranger(
    Solar~., data = rbind(train.data, test.data)[nrow(train.data) -11 + i:nrow(train.data) -1 + i], num.tree = 128, mtry = 2, importance = "impurity"
  )
  
  # method 1
  # for(idx in 1:data.pred %>% length()){
  #   get(data.pred[idx])[i] <- predict(model.ranger, data = test.data[i+idx-1,])
  # }
  
  # method 2
  data.pred01[i] <- predict(model.ranger, data = test.data[i,])
  data.pred02[i] <- predict(model.ranger, data = test.data[i+1,])
  data.pred03[i] <- predict(model.ranger, data = test.data[i+2,])
  data.pred04[i] <- predict(model.ranger, data = test.data[i+3,])
  data.pred05[i] <- predict(model.ranger, data = test.data[i+4,])
  data.pred06[i] <- predict(model.ranger, data = test.data[i+5,])
  data.pred07[i] <- predict(model.ranger, data = test.data[i+6,])
  data.pred08[i] <- predict(model.ranger, data = test.data[i+7,])
  data.pred09[i] <- predict(model.ranger, data = test.data[i+8,])
  data.pred10[i] <- predict(model.ranger, data = test.data[i+9,])
  data.pred11[i] <- predict(model.ranger, data = test.data[i+10,])
}

