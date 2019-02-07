# amazon_review
Goal: 
Predicting Amazon review scores based on text analysis

Features extraction: 
1. parsed the reviews into ngrams (1-3)
2. counted the term frequencies of ngrams
3. sentiment analysis based on ngrams
4. extra exploratory features: the ratio of question marks, the ratio of exclamation marks, and average sentence length

Model: 
Lasso regression with cross-validation
