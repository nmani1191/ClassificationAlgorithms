##----------------Machine Learning Algorithms for Classification----------------##

#---Data loading & Pre-Processing---------------------------

#Data : Tweets data

#Read the data
tweets<-read.csv("tweets.csv",stringsAsFactors=FALSE)
names(tweets)
dim(tweets)
names(tweets)[3]<-"reviews"

#Build a Text Corpus
library(tm)
tweets.corpus<-Corpus(VectorSource(tweets$reviews))
summary(tweets.corpus)
inspect(tweets.corpus[1:5]) #Inspecting elements in Corpus

#Data Transformations & Cleaning
tweets.corpus<-tm_map(tweets.corpus,tolower) #Converting to lower case
tweets.corpus<-tm_map(tweets.corpus,stripWhitespace) #Removing extra white space
tweets.corpus<-tm_map(tweets.corpus,removePunctuation) #Removing punctuations
tweets.corpus<-tm_map(tweets.corpus,removeNumbers) #Removing numbers
my_stopwords<-c(stopwords('english'),'available') #Can add more words apart from standard list
tweets.corpus<-tm_map(tweets.corpus,removeWords,my_stopwords)

#---Data loading & Pre-Processing---------------------------

#-------------------------Tag the sentiments using Jeffreybreen Algorithm---------------------------

#Read the text data
tweets.text<-tweets$reviews

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
#-------------------------Tag the sentiments---------------------------

#-------------------------Training and Test Data --------------------------------------

#Split the data into training and test(It was already classified based on random sampling)
train <- tweets[tweets$type=="train",]
test <- tweets[tweets$type=="test",]

prop.table(table(train$sentiment))
prop.table(table(test$sentiment))

#-------------------------Training and Test Data --------------------------------------

#----------------------------------Classification Algorithms -------------------------------

set.seed(2000)

#Loading Libraries
library(RTextTools)
library(e1071)

#building a dtm
matrix= create_matrix(tweets[,3], language="english",removeNumbers=TRUE, removePunctuation=TRUE, 
                      removeSparseTerms=0, 
                      removeStopwords=TRUE, stripWhitespace=TRUE, toLower=TRUE)

#Convert to a matrix data type
mat = as.matrix(matrix)

#Build the data to specify response variable, training set, testing set.
container = create_container(mat, as.numeric(tweets[,1]),
                             trainSize=1:2194, testSize=2195:3135,virgin=FALSE)

#Train the model
#Algorithms used to train the data : SVM,RF,TREE,MAXENT,BAGGING
models = train_models(container, algorithms=c("MAXENT", "SVM", "RF", "BAGGING", "TREE"))

#Test the model
results = classify_models(container, models)

#----------------------------------Classification Algorithms -------------------------------

#----------------------------------MODEL SUMMARY---------------------------------------

# Accuracy Table

# Confusion Matrix
table(as.numeric(as.factor(tweets[245:349,3])), results[,"FORESTS_LABEL"])
table(as.numeric(as.factor(tweets[245:349,3])), results[,"MAXENTROPY_LABEL"])
table(as.numeric(as.factor(tweets[245:349,3])), results[,"TREE_LABEL"])
table(as.numeric(as.factor(tweets[245:349,3])), results[,"BAGGING_LABEL"])
table(as.numeric(as.factor(tweets[245:349,3])), results[,"SVM_LABEL"])


# recall accuracy
recall_accuracy(as.numeric(as.factor(tweets[245:349,3])), results[,"FORESTS_LABEL"])
recall_accuracy(as.numeric(as.factor(tweets[245:349,3])), results[,"MAXENTROPY_LABEL"])
recall_accuracy(as.numeric(as.factor(tweets[245:349,3])), results[,"TREE_LABEL"])
recall_accuracy(as.numeric(as.factor(tweets[245:349,3])), results[,"BAGGING_LABEL"])
recall_accuracy(as.numeric(as.factor(tweets[245:349,3])), results[,"SVM_LABEL"])

analytics = create_analytics(container, results)
summary(analytics)
head(analytics@document_summary)

#----------------------------------MODEL SUMMARY---------------------------------------

#----------------------------------ENSEMBLE AGREEMENT----------------------------------

#Ensemble agreement = Simply refers to whether multiple make the same prediction concerning the the class of an event
analytics@ensemble_summary
#----------------------------------ENSEMBLE AGREEMENT----------------------------------

#--------------------------------CROSS VALIDATION----------------------------

# Cross validation is kind of technique to validate the model. Split the input data into required partitioned
# and generate model on 80% of eacn split and test the model in remaining 20% of split. This is more
# exhaustive model validation in compare with normal Foldout method (Train & Test data)

N=4
set.seed(2014)
cross_validate(container,N,"MAXENT")
cross_validate(container,N,"TREE")
cross_validate(container,N,"SVM")
cross_validate(container,N,"RF")
cross_validate(container,N,"BAGGING")

#--------------------------------CROSS VALIDATION----------------------------

#--------------------------------Final Predicted class choosen from multiple models-------------

#Based on which class is predicted by more number of model, that predicted class will consider as final class.

results1<-results[,c(1,3,5,7,9)]

#For each row
results1$majority=NA
for(i in 1:nrow(results1))
{
  #Getting the frequency distribution of the classifications 
  print(i)
  p<-data.frame(table(c(results1$MAXENTROPY_LABEL[i],results1$TREE_LABEL[i],results1$FORESTS_LABEL[i],
                        results1$SVM_LABEL[i],results1$BAGGING_LABEL[i])))
  #Choosing the classification that occurs maximum
  #Putting this value into the new column "majority"
  
  results1$majority[i]<-paste(p$Var1[p$Freq==max(p$Freq)])
  rm(p)
}

results1$majority<-as.numeric(results1$majority)
table(results1$majority)
#--------------------------------Final Predicted class choosen from multiple models-------------