
#----Document Classification using NAive Bayes Algorithm----------------------------------

#-----------------------Load data------------------------------------

#Read the data
analysis<-read.csv("tweets.csv",stringsAsFactors=FALSE)

#-----------------------Load data------------------------------------

#-------------------------Tag the sentiments using Jeffreybreen Algorithm---------------------------

#Read the text data
tweets.text<-analysis$text

#Read the dictionaries
pos = scan('positive-words.txt',what='character',comment.char=';')
neg = scan('negative-words.txt',what='character',comment.char=';')

#Adding words to dictionaries
pos[2007:2013]<-c("spectacular","everyday","better","top","thumbs","four","five")
neg[4784:4789]<-c("one","two","careful","sync","Beware","suck")

#Famous Jeffreybreen Algorithm to "Tag" sentiments to sentences
score.sentiment = function(sentences, pos.words, neg.words, .progress='none')
{
  require(plyr)
  require(stringr)
  
  #we got a vector of sentences. plyr will handle a list
  #or a vector as an "l" for us
  #we want a simple array of scores back, so we use
  #"l" + "a" + "ply" = "laply":
  scores = laply(sentences, function(sentence, pos.words, neg.words) {
    
    #clean up sentences with R's regex-driven global substitute, gsub():
    sentence = gsub('[[:punct:]]', '', sentence) #removes punctuations
    sentence = gsub('[[:cntrl:]]', '', sentence) #removes control characters
    sentence = gsub('\\d+', '', sentence) #removes digits
    
    #and convert to lower case:
    sentence = tolower(sentence)
    
    #split sentences into words. str_split is in the stringr package
    word.list = str_split(sentence, '\\s+')
    
    #sometimes a list() is one level of hierarchy too much
    words = unlist(word.list)
    
    #compare our words to the dictionaries of positive & negative terms
    pos.matches = match(words, pos.words)
    neg.matches = match(words, neg.words)
    
    #match() returns the position of the matched term or NA
    #we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    #and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
    
    return(score)
  }, pos.words, neg.words, .progress=.progress )
  
  scores.df = data.frame(score=scores, text=sentences)
  return(scores.df)
}

#Generate sentiment score for text
analysis<-score.sentiment(tweets.text, pos, neg, .progress="text")

names(analysis)
View(analysis)
str(analysis)

#Checking out overall sentiment
table(analysis$score)
mean(analysis$score)
hist(analysis$score)

#Tag the final sentiments based on the score
analysis$text<-as.character(analysis$text)
analysis$sentiment<-ifelse(analysis$score>0,"positive",
                           ifelse(analysis$score<0,"negative","neutral"))
table(analysis$sentiment)
#-------------------------Tag the sentiments using Jeffreybreen Algorithm---------------------------

#-------------------------Train & Test Datasets---------------------------
#Split the data into training and test
set.seed(2000)
sampling<-sort(sample(nrow(analysis), nrow(analysis)*.7))

length(sampling)

head(analysis)
names(analysis)
train_tweets = analysis[sampling,]
test_tweets = analysis[-sampling,]

prop.table(table(train_tweets$sentiment))
prop.table(table(test_tweets$sentiment))
#-------------------------Train & Test Datasets---------------------------

#-----------------------Naive Bayes algorithm ----------------------------

#Create induvidual matrices for training and testing datasets
mtrain<-as.matrix(train_tweets)
mtest<-as.matrix(test_tweets)

#Building Document term matrices for training and testing data
library(RTextTools)
library(e1071)
train_matrix= create_matrix(mtrain[,2], language="english",removeNumbers=TRUE, removePunctuation=TRUE, removeSparseTerms=0, 
                            removeStopwords=TRUE, stripWhitespace=TRUE, toLower=TRUE) 
test_matrix= create_matrix(mtest[,2], language="english",removeNumbers=TRUE, removePunctuation=TRUE, removeSparseTerms=0, 
                           removeStopwords=TRUE, stripWhitespace=TRUE, toLower=TRUE) 

#Input to naive bayes algorithm has to be a matrix with categorical values,not numeric

#Convert DTM to a 1/0 matrix
conversion<-function(A)
{
  A<-ifelse(A>0,1,0)
  A<-factor(A,levels=c(0,1),labels=c("No","Yes"))
  return(A)
}

library(tm)
View(inspect(train_matrix[1:10,1:10]))

mat_train<-apply(train_matrix,MARGIN=2,conversion)

mat_test<-apply(test_matrix,MARGIN=2,conversion)
View(mat_test[1:10,1:10])

#Train the model

#Input the training matrix and the Dependent Variable
classifier = naiveBayes(mat_train,as.factor(mtrain[,3]))

# Validation
predicted = predict(classifier,mat_test) #predicted
length(predicted)

#Model Performance Metrics
install.packages("gmodels")
library(gmodels)
install.packages("caret")
library(caret)

#Confusion Matrix
CrossTable(predicted,mtest[,3],prop.chisq=FALSE,prop.t=FALSE,dnn=c('predicted','actual'))

#accuracy of the model 
recall_accuracy(as.numeric(as.factor(test_tweets[,3])), as.numeric(predicted))

#Using Caret Package
confusionMatrix(predicted,mtest[,3])
#get values such as KAppa,Accuracy,Sensitivity/Recall,Specificity,PPV,NPV

#-----------------------Naive Bayes algorithm ----------------------------

