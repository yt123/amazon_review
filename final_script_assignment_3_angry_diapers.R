setwd("/Users/Apple/Desktop/big_data/archive")

library(tidyverse)
library(e1071)
library(tidytext)
library(glmnet)
library(dplyr)
library(beepr)
library(stringi)
library(syuzhet)

files <- unzip("Archive.zip")
amazon <- read_csv(files[2])

# ---------------------------------------------------------------
# split the reviews into sentences
reviews0 = amazon %>%
  mutate(id = row_number()) %>%
  unnest_tokens(sentence, review, token = "sentences", to_lower = FALSE) %>%
  count(id, name, rating, sentence)

#count the number of words in each sentence
amazon$sentence_word_count <- sapply(gregexpr("[[:alpha:]]+", amazon$review), length) #function(x) sum(x > 0))

#create new features: ratio of question marks, ratio of exclamation marks, average sentence length & sentiment scores
features_C = amazon %>%
  mutate(id = row_number()) %>%
  mutate(Rexcl = (stri_count_regex(review, paste("\\!", collapse="|")) / sentence_word_count)) %>% 
  mutate(Rques = (stri_count_regex(review, paste("\\?", collapse="|")) / sentence_word_count)) %>% 
  mutate(slength = sentence_word_count) #%>% 
  #mutate(sentiment = get_sentiment(review))
features_C = features_C[,5:9]
features_C[is.na(features_C)] <- 0

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

sentiments = reviews %>% 
  inner_join(get_sentiments("afinn"), by= c('token' = 'word')) %>% 
  group_by(id) %>%
  summarise(sentiment, n, fill = 0) %>% 
  mutate(token = 'sentiment')

sentiments = sentiments[c('id', 'token', 'tf', 'docfreq')]

# ---------------------------------------------------------------

features_short = features %>% 
  dplyr::select(id, token, tf, docfreq)

# sentiments = features_C %>% 
#   mutate(token = 'sentiment') %>% 
#   mutate(docfreq = 20) %>% 
#   rename(tf = sentiment)
# sentiments = sentiments[c('id', 'token', 'tf', 'docfreq')]

Rexcl = features_C %>% 
  mutate(token = 'Rexcl') %>% 
  mutate(docfreq = 20) %>% 
  rename(tf = Rexcl)
Rexcl = Rexcl[c('id', 'token', 'tf', 'docfreq')]

Rques = features_C %>% 
  mutate(token = 'Rques') %>% 
  mutate(docfreq = 20) %>% 
  rename(tf = Rques)
Rques = Rques[c('id', 'token', 'tf', 'docfreq')]

slength = features_C %>% 
  mutate(token = 'Slength') %>% 
  mutate(docfreq = 20) %>% 
  rename(tf = slength)
slength = slength[c('id', 'token', 'tf', 'docfreq')]

features_final = bind_rows(features_short, sentiments, Rexcl, slength, Rques)
head(features_final)
dim(features_final)

# -------------------------------------------------------------

format(object.size(features_final), units = "Gb")

dtm <- 
  filter(features_final, docfreq > 18) %>% 
  cast_sparse(row=id, column=token, value = tf)

format(object.size(dtm), units = "Gb")
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
             pred=predict(fit_lasso_lm, dtm[!trainidx, ], s = "lambda.min", type = "response")[,1])

pred_df = sample_submission %>%
  left_join(dtm_test_predictions) %>%
  mutate(Prediction = ifelse(Id %in% missing_test_ids, best_default_prediction, pred))

pred_df = pred_df[,c(1,2)]

write.csv(pred_df, file = "Output_final.csv", row.names = FALSE)