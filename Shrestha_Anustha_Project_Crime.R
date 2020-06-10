##Anustha Shrestha
##Prof. Rahnama Rad
##STA 9891 Project
##December 10,2019

#### Project ####
rm(list = ls())
cat()
library(ISLR)
library(MASS)
library(class)
library(glmnet)
library(randomForest)
library(ggplot2)
library(dplyr)
library(tidyr)
library(e1071)
library(caret)
library(gridExtra)
library(boot)

wholetime <- proc.time() #recording entire time 

#working directory and read csv
setwd("/Users/Anustha Shrestha/Documents/Baruch College/Fall 2019/STA 9891 Machine Learning/Project")
Crime<-read.csv("crime.csv")
Crime.X <-as.matrix(Crime[,-102])
Crime.Y <-as.factor(Crime[,102])

#Set seed and standardize data
set.seed(1)
Crime.X <-scale(Crime.X) #Standardized
n<-dim(Crime.X)[1]
nfold = 10
S=100 #Number of iterations
learnset = 0.5 #nlearn {0.5n,0.9n}

#defining errors vectors
svm.error = matrix(ncol = 2, nrow = S)
svm.error.cv = matrix(ncol = 1, nrow = S)
rf.error = matrix(ncol = 2, nrow = S)
glm.error = matrix(ncol = 2, nrow = S)
lasso.error = matrix(ncol = 2, nrow = S)
lasso.error.cv = matrix(ncol = 1, nrow = S)
ridge.error = matrix(ncol = 2, nrow = S)
ridge.error.cv = matrix(ncol = 1, nrow = S)
fp.svm.error = matrix(ncol=2, nrow=S)
fp.rf.error = matrix(ncol=2, nrow=S)
fp.glm.error = matrix(ncol=2, nrow=S)
fp.lasso.error = matrix(ncol=2, nrow=S)
fp.ridge.error = matrix(ncol=2, nrow=S)
fn.svm.error = matrix(ncol=2, nrow=S)
fn.rf.error = matrix(ncol=2, nrow=S)
fn.glm.error = matrix(ncol=2, nrow=S)
fn.lasso.error = matrix(ncol=2, nrow=S)
fn.ridge.error = matrix(ncol=2, nrow=S)

#defining time vectors
time.svm.cv =matrix(ncol =1, nrow=S)
time.svm.fit =matrix(ncol =1, nrow=S)
time.rf =matrix(ncol =1, nrow=S)
time.glm =matrix(ncol =1, nrow=S)
time.lasso.cv =matrix(ncol =1, nrow=S)
time.ridge.cv =matrix(ncol =1, nrow=S)
time.lasso.fit =matrix(ncol =1, nrow=S)
time.ridge.fit =matrix(ncol =1, nrow=S)

#Start Loop
for(i in 1:S){
#QN Randomly Split data into train and test set

random_order  =  sample(n)
n.train       =  floor(n*learnset)
n.test        =  n-n.train
trainSet      =  random_order[1:n.train]
testSet       =  random_order[(1+n.train):n]
Y             =  Crime.Y[trainSet]
X             =  Crime.X[trainSet,]
Y.test        =  Crime.Y[testSet]
X.test        =  Crime.X[testSet, ]
Y.train.dat   = as.data.frame(Y)
X.train.dat   = as.data.frame(X)
Y.test.dat   = as.data.frame(Y.test)
X.test.dat   = as.data.frame(X.test)

p=dim(X)[2]

#Removing imbalance in the data
imbalance     =     FALSE
if (imbalance == TRUE) {
  index.yis0      =      which(Y==0)  # idetify the index of  points with label 0
  index.yis1      =      which(Y==1) # idetify the index of  points with label 1
  n.train.1       =      length(index.yis1)
  n.train.0       =      length(index.yis0)
  if (n.train.1 > n.train.0) {     # we need more 0s in out training set, so we over sample with replacement
    more.train    =      sample(index.yis0, size=n.train.1-n.train.0, replace=TRUE)
  }         else {    # we need more 1s in out training set, so we over sample with replacement
    more.train    =   sample(index.yis1, size=n.train.0-n.train.1, replace=TRUE)
  }
  Y        =       as.factor(c(Y, Y[more.train])-1)
  X       =       rbind2(X, X[more.train,])
}


##SVM with radial kernel
ptm.svm.cv       = proc.time()
dat            = data.frame(X, Y)
tune.svm  =   tune(svm, X, Y, kernel = "radial",
                   ranges = list(cost = 10^seq(-2,2,length.out = 5),
                                 gamma = 10^seq(-2,2,length.out = 5) ))
svm.error.cv[i] = tune.svm$best.performance*100
ptm.svm.cv             =     proc.time() - ptm.svm.cv
time.svm.cv[i]        =     ptm.svm.cv["elapsed"]

ptm.svm.fit       = proc.time()
svm.fit =    svm(Y~. ,data=dat, kernel ="radial", cost =tune.svm$best.parameters[,1], gamma = tune.svm$best.parameters[,2] )
ptm.svm.fit           =     proc.time() - ptm.svm.fit
time.svm.fit[i]        =     ptm.svm.fit["elapsed"]

svm.pred.train = predict(svm.fit, X)
svm.error[i,1]<- mean(svm.pred.train!=Y)*100
fp.svm.error[i,1]<-mean(1 == svm.pred.train[Y==0])*100
fn.svm.error[i,1]<-mean(0 == svm.pred.train[Y==1])*100

svm.pred.test= predict(svm.fit, X.test)
svm.error[i,2]<- mean(svm.pred.test!=Y.test)*100
fp.svm.error[i,2]<-mean(1 == svm.pred.test[Y.test==0])*100
fn.svm.error[i,2]<-mean(0 == svm.pred.test[Y.test==1])*100


##Random Forest with m = sqrt(p) and 300 bootstrapped trees
ptm.rf = proc.time()
rf.model = randomForest(X, Y, mtry=round(sqrt(p)), ntree=300)
ptm.rf = proc.time()-ptm.rf
time.rf[i] = ptm.rf["elapsed"] 

rf.pred.train = predict(rf.model, X)
rf.error[i,1]<- mean(rf.pred.train!=Y)*100
fp.rf.error[i,1]<-mean(1 == rf.pred.train[Y==0])*100
fn.rf.error[i,1]<-mean(0 == rf.pred.train[Y==1])*100

rf.pred.test = predict(rf.model, X.test)
rf.error[i,2]<-mean(rf.pred.test!=Y.test)*100
fp.rf.error[i,2]<-mean(1 == rf.pred.test[Y.test==0])*100
fn.rf.error[i,2]<-mean(0 == rf.pred.test[Y.test==1])*100

##Logistic
dat            = data.frame(X, Y)
ptm.glm = proc.time()
glm.fitted     = glm(Y~., data=dat, family=binomial)
ptm.glm = proc.time()-ptm.glm
time.glm[i] = ptm.glm["elapsed"]

glm.prob.train =  predict(glm.fitted, X.train.dat, type="response" )
glm.pred.train = rep(0, dim(X)[1])
glm.pred.train[glm.prob.train>0.5]=1
glm.error[i,1]= mean(glm.pred.train!=Y)*100
fp.glm.error[i,1]<-mean(1 == glm.pred.train[Y==0])*100
fn.glm.error[i,1]<-mean(0 == glm.pred.train[Y==1])*100
  
glm.prob.test =  predict(glm.fitted, X.test.dat, type="response" )
glm.pred.test = rep(0, dim(X.test)[1])
glm.pred.test[glm.prob.test>0.5]=1
glm.error[i,2]= mean(glm.pred.test!=Y.test)*100
fp.glm.error[i,2]<-mean(1 == glm.pred.test[Y.test==0])*100
fn.glm.error[i,2]<-mean(0 == glm.pred.test[Y.test==1])*100

  
##LASSO
ptm.lasso.cv = proc.time()
lasso.cv      = cv.glmnet(X, Y, family = "binomial", alpha = 1, nfolds = nfold, type.measure="class")
lasso.error.cv[i]= min(lasso.cv$cvm)*100
ptm.lasso.cv = proc.time() - ptm.lasso.cv
time.lasso.cv[i] = ptm.lasso.cv["elapsed"]

ptm.lasso.fit = proc.time()
lam.lasso     = lasso.cv$lambda.min
lasso.fit     = glmnet(X, Y, lambda =lam.lasso, family = "binomial", alpha = 1)
ptm.lasso.fit = proc.time() - ptm.lasso.fit
time.lasso.fit[i] = ptm.lasso.fit["elapsed"]

lasso.pred.train =  predict(lasso.fit, s=lam.lasso, newx = X, type="class" )
lasso.error[i,1]= mean(lasso.pred.train!=Y)*100
fp.lasso.error[i,1]<-mean(1 == lasso.pred.train[Y==0])*100
fn.lasso.error[i,1]<-mean(0 == lasso.pred.train[Y==1])*100

lasso.pred.test =  predict(lasso.fit, s=lam.lasso, newx = X.test, type="class" )
lasso.error[i,2]= mean(lasso.pred.test!=Y.test)*100
fp.lasso.error[i,2]<-mean(1 == lasso.pred.test[Y.test==0])*100
fn.lasso.error[i,2]<-mean(0 == lasso.pred.test[Y.test==1])*100


##RIDGE
ptm.ridge.cv = proc.time()
ridge.cv      = cv.glmnet(X, Y, family = "binomial", alpha = 0, nfolds = nfold, type.measure="class")
ridge.error[i]= min(ridge.cv$cvm)*100
ptm.ridge.cv = proc.time()-ptm.ridge.cv
time.ridge.cv[i] = ptm.ridge.cv["elapsed"]

ptm.ridge.fit = proc.time()
lam.ridge     = ridge.cv$lambda.min
ridge.fit     = glmnet(X, Y, lambda =lam.ridge, family = "binomial", alpha = 0)
ptm.ridge.fit = proc.time()-ptm.ridge.fit
time.ridge.fit[i] = ptm.ridge.fit["elapsed"]

ridge.pred.train =  predict(ridge.fit, s=lam.ridge, newx = X, type="class" )
ridge.error[i,1]= mean(ridge.pred.train!=Y)*100
fp.ridge.error[i,1]<-mean(1 == ridge.pred.train[Y==0])*100
fn.ridge.error[i,1]<-mean(0 == ridge.pred.train[Y==1])*100

ridge.pred.test =  predict(ridge.fit, s=lam.ridge, newx = X.test, type="class" )
ridge.error[i,2]= mean(ridge.pred.test!=Y.test)*100
fp.ridge.error[i,2]<-mean(1 == ridge.pred.test[Y.test==0])*100
fn.ridge.error[i,2]<-mean(0 == ridge.pred.test[Y.test==1])*100

cat(sprintf("i=%1.f| Test:rf=%.2f| lasso=%.2f,ridge=%.2f|  ||| Train:rf=%.2f| lasso=%.2f,ridge=%.2f\n", i, rf.error[i,2], lasso.error[i,2],ridge.error[i,2],rf.error[i,1], lasso.error[i,1],ridge.error[i,1]))
}
#End Loop

#Time for Entire Loop
wholetime <- proc.time()- wholetime
time.entire = wholetime["elapsed"]

###Saving errors
ridge.error1<- ridge.error
lasso.error1<- lasso.error
glm.error1<- glm.error
rf.error1<- rf.error
svm.error1<- svm.error

#Mean Test Errors
ridgetesterror <- mean(ridge.error1[,2])
lassotesterror <- mean(lasso.error1[,2])
glmtesterror <- mean(glm.error1[,2])
rftesterror <- mean(rf.error1[,2])
svmtesterror <- mean(svm.error1[,2])

#Mean Time
cvridgetime<-mean(time.ridge.cv)
fitridgetime<-mean(time.ridge.fit)
cvlassotime<-mean(time.lasso.cv)
fitlassotime<-mean(time.lasso.fit)
fitglmtime<-mean(time.glm)
fitrftime<-mean(time.rf)
cvsvmtime<-mean(time.svm.cv)
fitsvmtime<-mean(time.svm.fit)
time.entire


#creating dataframe with all the train errors 
err.train           =     data.frame(c(rep("SVM", S),  rep("RF", S),  rep("Logistic", S), rep("Ridge", S),  rep("LASSO", S)) , 
                                     c(svm.error1[, 1], rf.error1[, 1], glm.error1[, 1], lasso.error1[, 1], ridge.error1[, 1]) )
colnames(err.train) =     c("Method","Error")

#Creating dataframe with test errors
err.test         =     data.frame(c(rep("SVM", S),  rep("RF", S),  rep("Logistic", S), rep("Ridge", S),  rep("LASSO", S)) , 
                                     c(svm.error1[, 2], rf.error1[, 2], glm.error1[, 2], lasso.error1[, 2], ridge.error1[, 2]) )
colnames(err.test) =     c("Method","Error")


#Creating dataframe with fp/fn train errors
err.train.fp         =     data.frame(c(rep("SVM", S),  rep("RF", S),  rep("Logistic", S), rep("Ridge", S),  rep("LASSO", S)) , 
                                  c(fp.svm.error[, 1], fp.rf.error[, 1], fp.glm.error[, 1], fp.lasso.error[, 1], fp.ridge.error[, 1]) )
colnames(err.train.fp) =     c("Method","Error")

err.train.fn         =     data.frame(c(rep("SVM", S),  rep("RF", S),  rep("Logistic", S), rep("Ridge", S),  rep("LASSO", S)) , 
                                      c(fn.svm.error[, 1], fn.rf.error[, 1], fn.glm.error[, 1], fn.lasso.error[, 1], fn.ridge.error[, 1]) )
colnames(err.train.fn) =     c("Method","Error")

#Creating dataframe with fp/fn test errors
err.test.fp         =     data.frame(c(rep("SVM", S),  rep("RF", S),  rep("Logistic", S), rep("Ridge", S),  rep("LASSO", S)) , 
                                     c(fp.svm.error[, 2], fp.rf.error[, 2], fp.glm.error[, 2], fp.lasso.error[, 2], fp.ridge.error[, 2]) )
colnames(err.test.fp) =     c("Method","Error")


err.test.fn         =     data.frame(c(rep("SVM", S),  rep("RF", S),  rep("Logistic", S), rep("Ridge", S),  rep("LASSO", S)) , 
                                  c(fn.svm.error[, 2], fn.rf.error[, 2], fn.glm.error[, 2], fn.lasso.error[, 2], fn.ridge.error[, 2]) )
colnames(err.test.fn) =     c("Method","Error")

#Save all the data into text files
write.table(err.train.fp, "TrainFPImbalance5.txt", sep='\t')
write.table(err.test.fp, "TestFPImbalance5.txt", sep='\t')
write.table(err.train.fn, "TrainFNImbalance5.txt", sep='\t')
write.table(err.test.fn, "TestFNImbalance5.txt", sep='\t')
write.table(err.train, "TrainError5Imbalance.txt", sep='\t')
write.table(err.test, "TestError5Imbalance.txt", sep='\t')

##########################   PLOTS    ##################################################

##Train and test errors 
p1 = ggplot(err.train)   +     aes(x=Method, y = Error, fill=Method) +   geom_boxplot()  +
  ggtitle("Train errors") + ylim(0,25)+
  theme( axis.title.x = element_blank(),
         plot.title          = element_text(size = 14, face = "bold"), 
         axis.title.y        = element_text(size = 12, face = "bold"),
         axis.text.x         = element_text(angle= 45, hjust= 1, size = 12, face = "bold"), 
         axis.text.y         = element_text(angle= 45, vjust = 0.7, size = 12, face = "bold"))
p1 <- p1+theme(legend.position ="none")



p2 = ggplot(err.test)   +     aes(x=Method, y = Error, fill=Method) +   geom_boxplot()  +
  ggtitle("Test errors") + ylim(0,25) +
  theme( axis.title.x = element_blank(),
         plot.title          = element_text(size = 14, face = "bold"), 
         axis.title.y        = element_text(size = 12, face = "bold"),
         axis.text.x         = element_text(angle= 45, hjust= 1, size = 12, face = "bold"), 
         axis.text.y         = element_text(angle= 45, vjust = 0.7, size = 12, face = "bold"))
p2 <- p2+theme(legend.position ="none")


##False Positive and False Negative Plots
p3 = ggplot(err.train.fp)   +     aes(x=Method, y = Error, fill=Method) +   geom_boxplot()  +
  ggtitle("False positive train errors") + ylim(0, 25)+
  theme( axis.title.x = element_blank(),
         plot.title          = element_text(size = 14, face = "bold"), 
         axis.title.y        = element_text(size = 12, face = "bold"),
         axis.text.x         = element_text(angle= 45, hjust= 1, size = 12, face = "bold"), 
         axis.text.y         = element_text(angle= 45, vjust = 0.7, size = 12, face = "bold"))
p3 <- p3+theme(legend.position ="none")

p4 = ggplot(err.test.fp)   +     aes(x=Method, y = Error, fill=Method) +   geom_boxplot()  +
  ggtitle("False positive test errors") + ylim(0,25)+
  theme( axis.title.x = element_blank(),
         plot.title          = element_text(size = 14, face = "bold"), 
         axis.title.y        = element_text(size = 12, face = "bold"),
         axis.text.x         = element_text(angle= 45, hjust= 1, size = 12, face = "bold"), 
         axis.text.y         = element_text(angle= 45, vjust = 0.7, size = 12, face = "bold"))
p4 <- p4+theme(legend.position ="none")

p5 = ggplot(err.train.fn)   +     aes(x=Method, y = Error, fill=Method) +   geom_boxplot()  +
  ggtitle("False Negative train errors") + ylim(0,25)+
  theme( axis.title.x = element_blank(),
         plot.title          = element_text(size = 14, face = "bold"), 
         axis.title.y        = element_text(size = 12, face = "bold"),
         axis.text.x         = element_text(angle= 45, hjust= 1, size = 12, face = "bold"), 
         axis.text.y         = element_text(angle= 45, vjust = 0.7, size = 12, face = "bold"))
p5 <- p5+theme(legend.position ="none")

p6 = ggplot(err.test.fn)   +     aes(x=Method, y = Error, fill=Method) +   geom_boxplot()  +
  ggtitle("False negative test errors") + ylim(0,25)+
  theme( axis.title.x = element_blank(),
         plot.title          = element_text(size = 14, face = "bold"), 
         axis.title.y        = element_text(size = 12, face = "bold"),
         axis.text.x         = element_text(angle= 45, hjust= 1, size = 12, face = "bold"), 
         axis.text.y         = element_text(angle= 45, vjust = 0.7, size = 12, face = "bold"))
p6 <- p6+theme(legend.position ="none")


grid.arrange(p1, p2,  ncol=2)
grid.arrange(p1,p3, p5,p2, p4, p6, ncol=3, nrow=2)

#10-fold CV curves for Lasso and Ridge Regression
#calculate beta ratios for ridge
m=25
tunetimeridge = proc.time()
ridge.cv1      = cv.glmnet(X, Y, family = "binomial", alpha = 0,type.measure="class")
lam.ridge1     =    exp(seq(log(max(ridge.cv1$lambda)),log(0.00001), -(log(max(ridge.cv1$lambda))-log(0.00001))/(m-1)))
ridge.cv1      = cv.glmnet(X, Y, lambda = lam.ridge1, family = "binomial", alpha = 0,type.measure="class")
tunetimeridge = proc.time()-tunetimeridge
tunetimeridge1 = tunetimeridge["elapsed"]

fittimeridge = proc.time()
ridge.fit.1     = glmnet(X, Y, lambda =ridge.cv1$lambda, family = "binomial", alpha = 0)
ridge.fit.0         =    glmnet(X, Y, lambda = 0, family = "binomial", alpha = 0)
n.lambdas     =   dim(ridge.fit.1$beta)[2]
ridge.beta.ratio    =    rep(0, n.lambdas)
for (j in 1:n.lambdas) {
  ridge.beta.ratio[j]   =   sqrt(sum((ridge.fit.1$beta[,j])^2)/sum((ridge.fit.0$beta)^2))
}
fittimeridge = proc.time()-fittimeridge
fittimeridge1 = fittimeridge["elapsed"]

#calculate beta ratios for lasso
m=25
tunetimelasso = proc.time()
lasso.cv1      =     cv.glmnet(X, Y, family = "binomial", alpha = 1,type.measure="class")
lam.lasso1     =    exp(seq(log(max(lasso.cv1$lambda)),log(0.00001), (log(0.00001) - log(max(lasso.cv1$lambda)))/(m-1)))
lasso.cv1      =     cv.glmnet(X, Y, lambda = lam.lasso1, family = "binomial", alpha = 1,type.measure="class")
tunetimelasso = proc.time()-tunetimelasso
tunetimelasso1 = tunetimelasso["elapsed"]

fittimelasso = proc.time()
lasso.fit.1     = glmnet(X, Y, lambda =lasso.cv1$lambda, family = "binomial", alpha = 1)
lasso.fit.0         =    glmnet(X, Y, lambda =0, family = "binomial", alpha = 1)
n.lambdas     =    dim(lasso.fit.1$beta)[2]
lasso.beta.ratio    =    rep(0, n.lambdas)
for (j in 1:n.lambdas) {
  lasso.beta.ratio[j]   =   sum(abs(lasso.fit.1$beta[,j]))/sum(abs(lasso.fit.0$beta))
}
fittimelasso = proc.time()-fittimelasso
fittimelasso1 = fittimelasso["elapsed"]


#combine lasso and ridge beta ratios and errors in the same data frame
eror           =     data.frame(c(rep("lasso", length(lasso.beta.ratio)),  rep("ridge", length(ridge.beta.ratio)) ), 
                                c(lasso.beta.ratio, ridge.beta.ratio) ,
                                c(lasso.cv1$cvm, ridge.cv1$cvm),
                                c(lasso.cv1$cvsd, ridge.cv1$cvsd))
colnames(eror) =     c("Method", "Ratio", "CV", "sd")

#plot lasso and ridge errors in same plot
eror.plot      =     ggplot(eror, aes(x=Ratio, y = CV, color=Method)) +   geom_line(size=1) 
eror.plot      =     eror.plot  + scale_x_log10()#(breaks = c(seq(0.1,2.4,0.2)))   
eror.plot      =     eror.plot  + theme(legend.text = element_text(colour="black", size=12, face="bold", family = "Courier")) 
eror.plot      =     eror.plot  + geom_pointrange(aes(ymin=CV-sd, ymax=CV+sd),  size=0.5,  shape=1)
eror.plot      =     eror.plot  + theme(legend.title=element_blank()) 
eror.plot      =     eror.plot  + scale_color_discrete(breaks=c("lasso", "ridge"))
eror.plot      =     eror.plot  + theme(axis.title.x = element_text(size=16),
                                        axis.title.y = element_text(size=16),
                                        axis.text.x  = element_text(angle=0, vjust=0.5, size=12,face="bold"),
                                        axis.text.y  = element_text(angle=0, vjust=0.5, size=12, face="bold")) 
eror.plot      =     eror.plot  + theme(plot.title = element_text(hjust = 0.5, vjust = -10, size=20, family = "Courier"))
eror.plot


#SVM Heatmap
ptm.svm.cv       = proc.time()
dat            = data.frame(X, Y)
tune.svm  =   tune(svm, X, Y, kernel = "radial",
                   ranges = list(cost = 10^seq(-2,2,length.out = 5),
                                 gamma = 10^seq(-2,2,length.out = 5) ))
svm.error.cv = tune.svm$best.performance*100
ptm.svm.cv             =     proc.time() - ptm.svm.cv
tunetimesvm       =     ptm.svm.cv["elapsed"] 

ptm.svm.fit       = proc.time()
svm.fit =    svm(Y~. ,data=dat, kernel ="radial", cost =tune.svm$best.parameters[,1], gamma = tune.svm$best.parameters[,2] )
ptm.svm.fit           =     proc.time() - ptm.svm.fit
fittimesvm        =     ptm.svm.fit["elapsed"] 
svmtune<- as.data.frame(tune.svm$performances)

gp <- ggplot(data=svmtune, aes(cost,gamma,fill=error))
gp <- gp + scale_x_log10(expand=c(0,0)) + scale_y_log10(expand=c(0,0))
gp <- gp + geom_tile()
gp <- gp + theme(axis.title.x = element_text(size=16),
                 axis.title.y = element_text(size=16),
                 axis.text.x  = element_text(angle=0, vjust=0.5, size=12, face="bold"),
                 axis.text.y  = element_text(angle=0, vjust=0.5, size=12, face="bold")) 
gp <- gp + theme(legend.text = element_text(colour="black", size=12, face="bold"))
gp <- gp + theme(legend.title = element_text(colour="black", size=16, face="bold"))
gp

#tune and fit times
tunetimeridge1
fittimeridge1
tunetimelasso1
fittimelasso1
tunetimesvm
fittimesvm

#Bar Plots for Coefficients for Logistic, Ridge, Lasso and RF
#plot for lasso coefficients
lassocoef <-as.matrix(coef(lasso.fit))
lassobar<-data.frame(c(rownames(lassocoef)), c(lassocoef))
colnames(lassobar) = c("Variable", "Lasso")
lassobar

lassobar1 <- ggplot(data=lassobar, aes(x=Variable, y=Lasso)) +
  geom_bar(stat="identity", color="#377EB8", fill="#377EB8", position=position_dodge())+
  theme_minimal()
lassobar1 <- lassobar1  + theme(axis.text.x=element_blank())
lassobar1 <- lassobar1 + theme(axis.title.y = element_text(size=12, face="bold"))
lassobar1 <- lassobar1  + theme(axis.title.x=element_blank())


#plot ridge coefficients
ridgecoef <-as.matrix(coef(ridge.fit))
ridgebar<-data.frame(c(rownames(ridgecoef)), c(ridgecoef))
colnames(ridgebar) = c("Variable", "Ridge")
ridgebar

ridgebar1 <- ggplot(data=ridgebar, aes(x=Variable, y=Ridge)) +
  geom_bar(stat="identity", color="black", fill="coral" , position=position_dodge())+
  theme_minimal()
ridgebar1 <- ridgebar1  + theme(axis.text.x=element_blank())
ridgebar1 <- ridgebar1  + theme(axis.text.x=element_blank())
ridgebar1 <- ridgebar1 + theme(axis.title.y = element_text(size=12, face="bold"))
ridgebar1 <- ridgebar1  + theme(axis.title.x=element_blank())


#plot random forest coefficients
rfvar <- as.matrix(rf.model$importance)
rfbar<-data.frame(c(rownames(rfvar)), c(rfvar))
colnames(rfbar) = c("Variable", "RF")
rfbar

rfbar1 <- ggplot(data=rfbar, aes(x=Variable, y=RF)) +
  geom_bar(stat="identity", color="black", fill="purple" , position=position_dodge())+
  theme_minimal()
rfbar1 <- rfbar1  + theme(axis.text.x=element_blank())
rfbar1 <- rfbar1 + theme(axis.title.y = element_text(size=12, face="bold"))
rfbar1 <- rfbar1 + theme(axis.title.x = element_text(size=12, face="bold"))

grid.arrange(lassobar1, ridgebar1, rfbar1, nrow=3)


#Sort the top 10 important variables
#Lasso top 10
lassonew<-lassobar[-1,]
lassonew<-arrange(lassonew, desc(abs(Lasso)))
lassonew1<-lassonew[1:10,]
lassobar2 <- ggplot(data=lassonew1, aes(x=reorder(Variable,abs(Lasso)), y=Lasso))+
  geom_bar(stat="identity", color="black", fill="#377EB8" , position=position_dodge())+coord_flip()
lassobar2 <- lassobar2 + theme(axis.title.y = element_blank())
lassobar2 <- lassobar2 + theme(axis.text.x = element_text(size=12, face="bold"))
lassobar2 <- lassobar2 + theme(axis.text.y = element_text(size=30, face="bold"))
lassobar2 <- lassobar2 + theme(axis.title.x = element_text(size=12, face="bold"))


#Ridge top 10
ridgenew<-ridgebar[-1,]
ridgenew<-arrange(ridgenew, desc(abs(Ridge)))
ridgenew1<-ridgenew[1:10,]
ridgebar2 <- ggplot(data=ridgenew1, aes(x=reorder(Variable,abs(Ridge)), y=Ridge))+
  geom_bar(stat="identity", color="black", fill="coral" , position=position_dodge())+coord_flip()
ridgebar2 <- ridgebar2 + theme(axis.title.y = element_blank())
ridgebar2 <- ridgebar2 + theme(axis.text.x = element_text(size=12, face="bold"))
ridgebar2 <- ridgebar2 + theme(axis.text.y = element_text(size=30, face="bold"))
ridgebar2 <- ridgebar2 + theme(axis.title.x = element_text(size=12, face="bold"))

#Random Forest top 10
rfnew<-arrange(rfbar, desc(abs(RF)))
rfnew1<-rfnew[1:10,]
rfbar2 <- ggplot(data=rfnew1, aes(x=reorder(Variable,RF), y=RF))+
  geom_bar(stat="identity", color="black", fill="purple" , position=position_dodge())+coord_flip()
rfbar2 <- rfbar2 + theme(axis.title.y = element_blank())
rfbar2 <- rfbar2 + theme(axis.text.x = element_text(size=12, face="bold"))
rfbar2 <- rfbar2 + theme(axis.text.y = element_text(size=30, face="bold"))
rfbar2 <- rfbar2 + theme(axis.title.x = element_text(size=12, face="bold"))

ridgebar2
lassobar2 
rfbar2
