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
DEFAULT_PATH <- "C:\\Users\\student\\Desktop\\solar\\Solar_radiation_prediction_model\\HanTaegyu\\transfer_learning"
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
  train.data <- train.data %>% rbind(orignal.data[[idx]][1:test.index,])
  print(value.names[idx])
  print(orignal.data[[idx]][1:test.index,] %>% nrow())
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

# Train 데이터 생성
test.data <- orignal.data$Seoul %>% 
  filter(Year < 2014) %>% 
  arrange("Year", "Month", "Day", "Hour")

train.data %>% dim()
# [1] 156275      9 
test.data %>% dim()
# [1] 12052       9

# ranger model
ranger.model <- ranger::ranger(Solar ~ ., 
                train.data, 
                mtry = 5,  # 노드수 분리할 수 있는 노드의 변수(의년이 피셜 노드 분기 수)
                importance = "impurity")  # 지니계수 

# test 확인
# # predict.list 생성
# predict.list <- list()
# for (idx in seq(11)) {
#   predict.list[[idx]] <- data.frame()
# }
# names(predict.list) <- paste0("number.", seq(11))

# future::plan(multisession, workers = 7)

custom.predict <- function(idx, jdx) {
  predict.result <- stats::predict(ranger.model, 
                                   data = test.data[idx + jdx - 1,] %>% select(-Solar))
  
  return(data.frame(
    predictions = predict.result$predictions,
    error = test.data[idx + jdx, c("Solar")] - predict.result$predictions
  ))
}

term <- round((test.data %>% nrow() - 11) / 6)
1: term
term + 1: 




split.data <- split(seq((test.data %>% nrow() - 11)),
                ceiling(seq((test.data %>% nrow() - 11))/2007))

for (i in seq(split.data %>% length())) {
  (split.data[[i]] %>% length() %% 11) %>% print()
}
# [1] 5
# [1] 5
# [1] 5
# [1] 5
# [1] 5
# [1] 4

# PC1
1:((184 * 11) * 1) %>% min()
1:((184 * 11) * 1) %>% max()
# [1] 1:2024

# PC2
((184 * 11) * 1):((184 * 11) * 2) %>% min()
((184 * 11) * 1):((184 * 11) * 2) %>% max()
# [1] 2024,4048

# PC3
((184 * 11) * 2):((184 * 11) * 3) %>% min()
((184 * 11) * 2):((184 * 11) * 3) %>% max()
# [1] 4048,6072

# PC4
((184 * 11) * 3):((184 * 11) * 4) %>% min()
((184 * 11) * 3):((184 * 11) * 4) %>% max()
# [1] 6072,:8096

# PC5
((184 * 11) * 4):((184 * 11) * 5) %>% min()
((184 * 11) * 4):((184 * 11) * 5) %>% max()
# [1] 8096,:10120

# PC6
((184 * 11) * 5):(test.data %>% nrow() - 11) %>% min()
((184 * 11) * 5):(test.data %>% nrow() - 11) %>% max()
# [1] 10120,12041


predict.value2 <- future.apply::future_lapply(
  X = seq(1, 2),
  future.packages = c("stats", "ranger"),
  function(idx) {
    idx.rep <- rep(idx, 11)
    jdx.rep <- seq(1, 11)
    map2(idx.rep, jdx.rep, custom.predict) %>% list.rbind()
  }
)

save.image(predict.value2, )

