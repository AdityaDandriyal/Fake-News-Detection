**FAKE NEWS DETECTION using Machine Learning Algorithms**

**DESCRIPTION**

Fake news is false or misleading information presented as news. Fake news often has the aim of damaging the reputation of a person or entity, or making money through advertising revenue. The
prevalence of fake news has increased with the recent rise of social media and this misinformation is gradually seeping its way into the mainstream media. Fake news can reduce the impact of real news by
competing with and also carry the potential to undermine trust in serious media coverage. Thus a need arises to identify whether a news is fake or not. With this project we aim to do the same.

**OBJECTIVE**

This is a Binary Classification Problem, where the objective is to classfiy whether a news article is fake or real. Label 0 corresponds to Real News whereras Label 1 corresponds to Fake News.

**DATASET**

The dataset used was downloaded from kaggle.It has 20k data points with the following attributes: 
id: unique id for a news article , title: the title of a news article, author: author of the news article, 
text: the text of the article; could be incomplete, label: a label that marks the article as potentially unreliable ( 1: fake , 0: real).

**DATA-PREPROCESSING**

Firstly we dealt with the null values in the dataset by replacing them with an empty string. 
Then we merged the columns ‘title’ and ‘author’ into a single column and used this column only as we proceeded further.
Since the data is textual in nature so stemming, lemmatization and stopwords removal techniques were used to clean the data.

**VECTORIZATION TECHNIQUES**

Converts data from its raw textual format into vectors of real numbers so we can feed it to a machine learning model. The vectorization techniques used by us are:

• **Bag of Words:** in this method a sparse matrix is created for the input, out of the frequency of vocabulary words. In this sparse matrix each row is a document and each column
represents word in a corpus.

• **TF-IDF vectorizer:** in this method a sparse matrix is created for the input, out of the tfidf values of vocabulary words. In this sparse matrix each row is a document and each
column represents words in a corpus.

• **Word2Vec:** This approach uses the power of a simple neural network to generate word
embeddings. Each word is represented as a 100 dimensional vector and words having similar meaning exist in close proximity to each other in the 100 dimensional hyperspace.
Each document is represented as a vector by taking the mean of all the words in the document in vector format.

**FEATURE REDUCTION**

In order to drop some columns and reduce the matrix dimensionality of the
sparse matix in TF-IDF and Bag of Words we carry out some feature selection using Chi-Squared
Test to determine whether a feature and the target are independent and to keep only the features
with a certain p-value from the Chi-Square test.


**MODEL TRAINING**

We trained various models namely Logistic Regression, Decision Tree Classifier,
Random Forest Classifier, Multinomial Naïve Bayes Classifier, K-Nearest -Neighbour and
Support Vector Classifier.
For dataset obtained through word2vec vectorization, instead of Multinomial we make use of
Gaussian Naïve Bayes Classifier as Multinomial NB Classifier doesn’t take negative values.

**MODEL SELECTION**

We used K-fold cross validation with scoring parameter as accuracy to test the
models and find the accuracy of all the models. Further we are using Hypothesis testing (using
5x2 CV Paired T Test) on the top two performers of the K-fold cross validation to verify the claim.

**EXPERIMENTAL RESULTS**

![image](https://github.com/user-attachments/assets/6375e1c9-ed9d-43de-be89-5c8963d46e0e)


**CONCLUSION**

Analyzing the text data is critical. This project is focused on applying vectorization techniques
such as bag of words, TF-IDF and word2vec to preprocess and vectorize text and evaluate its
effectiveness by running them through Logistic regression, Decision Tree, Random Forest, Multinomial
Naïve Bayes, K-Nearest Neighbors and SVM classifiers.
The result of the above approaches showed that bag of words vectorization technique with Decision Tree
or SVC (Linear kernel) showed the highest accuracy, but since Bag of words vectorization with
features reduction using chi-square test with SVC (Linear kernel) classifier also has almost the same
accuracy, hence considering the benefits of feature reduction this approach comes out to be the better one
for text classification.
