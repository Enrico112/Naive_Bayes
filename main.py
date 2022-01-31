import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

df = pd.read_csv('spam.csv')
df.head()

# group by cat (ham means non-spam)
df.groupby('Category').describe()

# convert category from string to bin
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)

# split train and test dfs
X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam, test_size=.2)

# create count vectorizer to count frequency of unique words in message
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()

# instance multinomial naive baysian obj
model = MultinomialNB()
# must fit X_ count, not array
model.fit(X_train_count, y_train)

# test model with made-up emails
emails = ["Hey, wdo you want to watch football together?",
          "Discount on parking, exclusive offer"]
emails_count = v.transform(emails)
model.predict(emails_count)

# get unique words count in X_test
X_test_count = v.transform(X_test.values)
# test accuracy with X_test
model.score(X_test_count, y_test)

# Redo with pipeline

# create pipeline
pipe = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
# fit pipeline
pipe.fit(X_train, y_train)
# check score
pipe.score(X_test, y_test)