# --------------------------------------------- #
# 2022-10-03 : 순천향대학교 전이학습 모델
# 
# --------------------------------------------- #
rm(list = ls())

pc_number <- 2
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
DEFAULT_PATH <- "C:\\Users\\student\\Desktop\\solar\\Solar_radiation_prediction_model\\HanTaegyu\\transfer_learning"
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



loopnum.cut.11 <- loopnum - 11

for(idx in (1:(data.pred %>% length()))){
  assign(data.pred[idx], numeric(loopnum.cut.11))
}


# 데이터 나누기
multi_pc_data_index <- split(seq(loopnum.cut.11), ceiling(seq(loopnum.cut.11)/loopnum.cut.11 %/% 6))

for (i in 1:6) {
  paste0(
    multi_pc_data_index[[i]] %>% min() %>% as.character(),
    ":",
    multi_pc_data_index[[i]] %>% max() %>% as.character(),
    "     PC",
    i
  ) %>% print()
}


# [1] "1:1336     PC1"
# [1] "1337:2672     PC2"
# [1] "2673:4008     PC3"
# [1] "4009:5344     PC4"
# [1] "5345:6680     PC5"
# [1] "6681:8016     PC6"

if (pc_number == 1) {
  run_seq <- 1:1336  
} else if (pc_number == 2) {
  run_seq <- 1337:2672  
} else if (pc_number == 3) {
  run_seq <- 2673:4008
} else if (pc_number == 4) {
  run_seq <- 4009:5344  
} else if (pc_number == 5) {
  run_seq <- 5345:6680
} else if (pc_number == 5) {
  run_seq <- 6681:8016
}

# 11시점을 예측하기 위해서 11 만큼만 봐줌
# loopnum
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
  data.pred02[i] <- predict(model.ranger, data = test.data[i+ 12,]) %>% as.numeric()
  data.pred03[i] <- predict(model.ranger, data = test.data[i+ 13,]) %>% as.numeric()
  data.pred04[i] <- predict(model.ranger, data = test.data[i+ 14,]) %>% as.numeric()
  data.pred05[i] <- predict(model.ranger, data = test.data[i+ 15,]) %>% as.numeric()
  data.pred06[i] <- predict(model.ranger, data = test.data[i+ 16,]) %>% as.numeric()
  data.pred07[i] <- predict(model.ranger, data = test.data[i+ 17,]) %>% as.numeric()
  data.pred08[i] <- predict(model.ranger, data = test.data[i+ 18,]) %>% as.numeric()
  data.pred09[i] <- predict(model.ranger, data = test.data[i+ 19,]) %>% as.numeric()
  data.pred10[i] <- predict(model.ranger, data = test.data[i+ 20,]) %>% as.numeric()
  data.pred11[i] <- predict(model.ranger, data = test.data[i+ 21,]) %>% as.numeric()
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