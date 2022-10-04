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
# --------------------------------------------- #


# --------------------------------------------- # 
# 데이터 불러오기

# 프로젝트 경로
DEFAULT_PATH <- "C:\\Users\\gksxo\\Desktop\\Folder\\github\\Solar_radiation_prediction_model\\HanTaegyu\\transfer_learning"
setwd(DEFAULT_PATH)
getwd()
# [1] "C:/Users/student/Desktop/solar/Solar_radiation_prediction_model/HanTaegyu/transfer_learning"

DATA_FOLDER_NAME <- "\\data"  # 데이터 폴더 이름
DATA_PATH <- paste0(DEFAULT_PATH, DATA_FOLDER_NAME) # 데이터 폴더 경로
files <-list.files(DATA_PATH) # 데이터 파일 이름들

# 파일 경로들
FILES_PATH <- paste0(DATA_PATH, paste0("\\", files))


# 데이터 읽기
orignal.data <- list()
for(idx in FILES_PATH %>% length() %>% seq()) {      
  orignal.data[[idx]] <- read.csv(file = FILES_PATH[idx],
                                  header = TRUE)
}

value.names <- str_remove(files, ".csv")
names(orignal.data) <- value.names


# 데이터 크기 확인
for(idx in FILES_PATH %>% length() %>% seq()) {
  ls(orignal.data)[idx] %>% print()   # 변수이름
  orignal.data[[idx]] %>% dim() %>% print()  # 데이터 프레임 크기
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

orignal.data$Busan %>% names()
# [1] "Year"  "Month" "Day"   "Hour"  "Temp"  "Humi"  "WS"    "WD"    "Solar"


# 데이터 시간순으로 정렬
for(idx in FILES_PATH %>% length() %>% seq()) {
  orignal.data[[idx]] <- orignal.data[[idx]] %>% 
    arrange("Year", "Month", "Day", "Hour")
}

# 첫번째 데이터()
test.index <- orignal.data$Busan %>% 
  filter(Year < 2019) %>% 
  nrow()
# [1] 31255

# 2011 ~ 2018년 데이터 추출
# 서울 제외한 데이터
train.data <- data.frame()
for(idx in FILES_PATH %>% length() %>% seq()) {
  if (value.names[idx] == "Seoul") { next }
  
  train.data %>%
    rbind(orignal.data[[idx]] %>% filter(Year < 2019))
  print(value.names[idx])
  print(orignal.data[[idx]] %>% filter(Year < 2019) %>% nrow())
}
# [1] "Busan"
# [1] 31255
# [1] "Daegu"
# [1] 31255
# [1] "Daejeon"
# [1] 31255
# [1] "Gwangju"
# [1] 31255
# [1] "Incheon"
# [1] 31255