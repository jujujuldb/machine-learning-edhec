library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(knitr)
library(corrplot)
library(listings)
library(ggplot2)
library(dplyr)
library(tidyverse)





#Data Preparation 
set.seed(1)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

traindata <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testdata <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))


inTrain <- createDataPartition(y=traindata$classe, p=0.7, list=F)
traindata1 <- traindata[inTrain, ]
traindata2 <- traindata[-inTrain, ]
# we remove the variables with nearly zero variance
nzv <- nearZeroVar(traindata1)
traindata1 <- traindata1[, -nzv]
traindata2 <- traindata2[, -nzv]

# we remove the variables that are mostly NA
mostlyNA <- sapply(traindata1, function(x) mean(is.na(x))) > 0.95
traindata1 <- traindata1[, mostlyNA==F]
traindata2 <- traindata2[, mostlyNA==F]

# we remove the variables that don't make intuitive sense for prediction cad the first five variables
traindata1 <- traindata1[, -(1:5)]
traindata2 <- traindata2[, -(1:5)]

corMatrix <- cor(traindata1[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))

set.seed(1)
decisionTreeMod1 <- rpart(classe ~ ., data=trainData, method="class")
fancyRpartPlot(decisionTreeMod1)

#Predictions with classification Trees
set.seed(1)
ctrlfit1 <- rpart(classe ~ ., data=traindata1, method="class")
fancyRpartPlot(ctrlfit1)

predictionsA1 <- predict(ctrlfit1, traindata2, type = "class")
cmtree <- confusionMatrix(predictionsA1, traindata2$classe)
cmtree

plot(cmtree$table, col = cmtree$byClass, main = paste("Decision Tree Confusion Matrix: Accuracy =", round(cmtree$overall['Accuracy'], 4)))

#Prediction with random forest 
set.seed(1)
ctrlfit2 <- randomForest(classe ~ ., data=traindata1)
predictionB1 <- predict(ctrlfit2, traindata2, type = "class")
cmrf <- confusionMatrix(predictionB1, traindata2$classe)
cmrf

plot(ctrlfit2)

plot(cmrf$table, col = cmtree$byClass, main = paste("Random Forest Confusion Matrix: Accuracy =", round(cmrf$overall['Accuracy'], 4)))

#Prediction with Generalized Boosted Regression
set.seed(1)
ctrlgbr <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1)

gbmFit1 <- train(classe ~ ., data=traindata1, method = "gbm",
                 trControl = ctrlgbr,
                 verbose = FALSE)


gbmFinMod1 <- gbmFit1$finalModel

gbmPredTest <- predict(gbmFit1, newdata=traindata2)
gbmAccuracyTest <- confusionMatrix(gbmPredTest, traindata2$classe)
gbmAccuracyTest

plot(gbmFit1, ylim=c(0.9, 1))


#Predicting Results on the Test Data

predictionFinal <- predict(ctrlfit2, testing, type = "class")
predictionFinal

# Write the results to a text file for submission
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}