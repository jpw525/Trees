##Fitting regression tree
library(MASS)
library(ISLR)
library(tree)
attach(Boston)
names(Boston)
set.seed(1)
#create training set
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston = tree(medv~.,Boston, subset=train)
summary(tree.boston)

#---in summary you will see: ----
#variables actually used in tree construction: 
#Number of terminal nodes:
#Residual mean deviance: === in a regression tree deviance is the sume of squared errors for the tree
#Distribution of residuals:

#plot the tree
plot(tree.boston)
#add values to the tree 
text(tree.boston, pretty=0)

#cross validation of the tree to see what tree is selected
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size, cv.boston$dev, type = 'b')

## If you want to prune the tree
prune.boston=prune.tree(tree.boston,best=5)
plot(prune.boston)
text(prune.boston, pretty=0)

#pick your tree (use cv or pruned), below is for an unpruned tree
yhat=predict(tree.boston, newdata=Boston[-train,])
boston.test=Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0,1)
mean((yhat-boston.test)^2) #gives you the MSE associated with the regression tree


####bagging and random forest
library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston, subset=train, mtry=13, importance=TRUE)
bag.boston #bagging is simply a special case of a random forest with m=p
#mtry=13 indicates that all 13 predictors should be considered for each split of the tree

#see how well this bagged model perform on the test set?
yhat.bag=predict(bag.boston, newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
#get the test set MSE
mean((yhat.bag-boston.test)^2)

#change number of cheese grown with ntree arguement
bag.boston=randomForest(medv~.,data=Boston, subset = train, mtry=13,ntree=25)
yhat.bag=predict(bag.boston, newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)


###Random Forest
#is the same as above but we use a smaller mtry = #
#by default randomForest() uses p/3 variables for regression trees
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=T)
yhat.rf=predict(rf.boston,newdata=Boston[-train,])
#get test set MSE
mean((yhat.rf-boston.test)^2)
#use importance function to view the importance of each variable
importance(rf.boston)
#plot the importance
varImpPlot(rf.boston)

##Boosting
library(gbm)
set.seed(1)
boost.boston=gbm(medv~.,data=Boston[train,], distribution="gaussian",n.trees=5000,interaction.depth=4)
#use summary function to produce relative influence plot and outputs of relative influence statistics
summary(boost.boston)
#produce partial dependence plots for the influencial variables
##these plots illustrate the marginal effect of the selected variables on the response after intergrating out the other variables
par(mfrow=c(1,2))
plot(boost.boston, i="rm")
plot(boost.boston, i="lstat")
#now use the boosted model to predict Y on the test set
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
#get test MSE
mean((yhat.boost-boston.test)^2)

###references

#creds to this book
#Introduction to Applied Statistical Learning with Applications in R by James, Witten, Hastie & Tibshirani
#http://www-bcf.usc.edu/~gareth/ISL/ISLR%20First%20Printing.pdf






