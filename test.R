setwd("/Users/Apple/Desktop/big_data/archive")

library(tidyverse)
library(e1071)
library(tidytext)
library(glmnet)
library(dplyr)
library(beepr)

files <- unzip("Archive.zip")
amazon <- read_csv(files[2])

# ---------------------------------------------------------------
#split the reviews into sentences
reviews0 = amazon %>%
  mutate(id = row_number()) %>%
  unnest_tokens(sentence, review, token = "sentences", to_lower = FALSE) %>%
  count(id, name, rating, sentence)

#count the number of words in each sentence
reviews0$sentence_word_count <- sapply(gregexpr("[[:alpha:]]+", reviews0$sentence), function(x) sum(x > 0))

#create new features: ratio of question marks, ratio of exclamation marks, average sentence length
features0 = reviews0 %>%
  group_by(id) %>%
  mutate(questionRatio = length(grep("\\?", sentence))/n) %>%
  mutate(ExclamationRatio = length(grep("\\!", sentence))/n) %>%
  mutate(nwordsperson = sum(sentence_word_count) / sum(n))

features0 = features0[,c(1,7,8,9)]
features0 = unique(features0)
features0 = na.omit(features0)

# ---------------------------------------------------------------
# getting the sentiment score for each person
library(syuzhet)

reviews_sentiment = amazon %>%
  mutate(id = row_number()) %>%
  mutate(sentiment = get_sentiment(review))

sentiments = reviews_sentiment[,c(4,5)]

# ---------------------------------------------------------------

trainidx <- !is.na(amazon$rating)
table(trainidx) # so there are 153531 training samples and the rest is for testing

# ---------------------------------------------------------------
# creating ngrams
reviews = amazon %>% 
  mutate(id = row_number()) %>%
  unnest_tokens(token, review, token = "ngrams", n = 3, n_min = 1) %>%
  count(id, name, rating, token)

head(reviews)

# ---------------------------------------------------------------

# sw <- reviews %>%
#   inner_join(get_stopwords(), by = c(token='word')) %>%
#   group_by(id) %>%
#   mutate(N = sum(n))
# 
# head(sw)

# ---------------------------------------------------------------
# getting the term frequencies for ngrams
features = 
  reviews %>%
  group_by(id) %>%
  mutate(nwords = sum(n)/3 + 1) %>% # the number of tokens per document
  group_by(token) %>%
  mutate(
    docfreq = n(), # number of documents that contain the token
    tf = n / nwords  # relative frequency of token within the given document
  ) %>%
  ungroup()

head(features)

# ---------------------------------------------------------------
# selecting id, token, tf, docfreq from features
features_short = features %>% 
  dplyr:: select(id, token, tf, docfreq)


sentiments = sentiments %>% 
  mutate(token = 'sentiment') %>% 
  mutate(docfreq = 20) %>% 
  rename(tf = sentiment)
sentiments = sentiments[c('id', 'token', 'tf', 'docfreq')]

QRatio = features0 %>% 
  dplyr::select(id, questionRatio) %>% 
  mutate(token = 'QRatio') %>% 
  mutate(docfreq = 20) %>% 
  rename(tf = questionRatio)
QRatio = QRatio[c('id', 'token', 'tf', 'docfreq')]

ERatio = features0 %>% 
  dplyr::select(id, ExclamationRatio) %>% 
  mutate(token = 'ERatio') %>% 
  mutate(docfreq = 20) %>% 
  rename(tf = ExclamationRatio)
ERatio = ERatio[c('id', 'token', 'tf', 'docfreq')]

slength= features0 %>% 
  dplyr::select(id, nwordsperson) %>% 
  mutate(token = 'sentence_length') %>% 
  mutate(docfreq = 20) %>% 
  rename(tf = nwordsperson)
slength= slength[c('id', 'token', 'tf', 'docfreq')]

features_final = bind_rows(features_short, sentiments, QRatio, ERatio, slength)
head(features_final)


# ---------------------------------------------------------------


# c(rows = nrow(amazon), cols = features$token %>% unique %>% length)

# ---------------------------------------------------------------

format(object.size(features), units = "Gb")

# matrix_features_final2 = data.matrix(features_final2[,c(1, 3, 5:12)])
# 
# df.features = as.data.frame(as.table(matrix_features_final2)) %>% rename(row = Var1, col=Var2, value = Freq) 
# head(df.features) # the same matrix stored as a data frame
# 
# format(object.size(df.features), units = "Gb")
# 
# sparse.features = filter(df.features, value != 0) 
# head(sparse.features) # a sparse represtation of M
# tail(sparse.features)
# 
# format(object.size(sparse.features), units = "Gb")

# -------------------------------------------------------------

dtm <- 
  filter(features_final, docfreq > 18) %>% 
  cast_sparse(row=id, column=token, value = tf)

#format(object.size(dtm), units = "Gb")

dim(dtm)

# -------------------------------------------------------------

used_rows = as.integer(rownames(dtm))
used_amazon = amazon[used_rows, ]
trainidx = trainidx[used_rows]
table(trainidx)

file_name_id <- unzip("Archive.zip")
sample_submission = read_csv(file_name_id[1], col_types = cols(col_character(), col_double()))

library(glmnet)
y = used_amazon$rating
y[y < 4] <- 0
y[y > 3] <- 1
fit_lasso_lm = cv.glmnet(dtm[trainidx,], y[trainidx], alpha = 1, type.measure = "auc", family = "binomial")


# used_rows computed earlier contains all the indices of reviews used in dtm
all_test_reviews = which(is.na(amazon$rating))
missing_test_ids = setdiff(used_rows, all_test_reviews)

best_default_prediction = mean(y[trainidx]) # best prediction if now review features are available
cat("best baseline prediction:", best_default_prediction,"\n")

dtm_test_predictions = 
  data.frame(Id = as.character(used_rows[!trainidx]), 
             pred=predict(fit_lasso_lm, dtm[!trainidx, ], s = "lambda.min", type = "response")[,1]
  )


pred_df = sample_submission %>%
  left_join(dtm_test_predictions) %>%
  mutate(Prediction = ifelse(Id %in% missing_test_ids, best_default_prediction, pred))

pred_df = pred_df[,c(1,2)]

write.csv(pred_df, file = "Output_oct9.csv", row.names = FALSE)
