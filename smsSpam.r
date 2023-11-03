# Load required packages
library(tm)
library(SnowballC)

library(wordcloud)
library(e1071)
library(gmodels)

# Read SMS data from CSV
sms_raw <- read.csv(file.choose(), sep = ",", header = TRUE, stringsAsFactors = FALSE)

# Convert 'type' column to factor
sms_raw$type <- factor(sms_raw$type)

# Create a corpus from the 'text' column
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

# Preprocessing steps
sms_corpus <- tm_map(sms_corpus, content_transformer(tolower))
sms_corpus <- tm_map(sms_corpus, removeNumbers)
sms_corpus <- tm_map(sms_corpus, removeWords, stopwords())
sms_corpus <- tm_map(sms_corpus, removePunctuation)
sms_corpus <- tm_map(sms_corpus, stemDocument)
sms_corpus <- tm_map(sms_corpus, stripWhitespace)

# Create a document-term matrix
sms_dtm <- DocumentTermMatrix(sms_corpus)

# Split data into training and testing sets
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

# Extract labels for training and testing sets
sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

# Generate word cloud
wordcloud(sms_corpus, min.freq = 50, random.order = FALSE)

# Identify frequently used words
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

# Filter document-term matrix with frequent words
sms_dtm_freq_train <- sms_dtm_train[, sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[, sms_freq_words]

# Convert counts to binary values
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

# Train a naive Bayes classifier
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

# Make predictions on the test set
sms_test_pred <- predict(sms_classifier, sms_test)

# Evaluate the classifier's performance
table(sms_test_pred, sms_test_labels)
CrossTable(sms_test_pred, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c('predicted', 'actual'))