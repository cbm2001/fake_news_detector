import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./news.csv')
# df.head()
labels = df.authenticity.head()

df = df.drop(["article_no","title"], axis=1)

# split the data set into train and test 
x_train,x_test,y_train,y_test=train_test_split(df['text'], df['authenticity'], test_size=0.3, random_state=7)

# Initialize a TfidfVectorizer
vectorizer=TfidfVectorizer()

x_vector_train=vectorizer.fit_transform(x_train) 
x_vector_test=vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pass_agg=PassiveAggressiveClassifier(max_iter=50)
pass_agg.fit(x_vector_train,y_train)

# Predict on the test set and calculate accuracy
pred=pass_agg.predict(x_vector_test)
score=accuracy_score(y_test,pred)
print(f'Accuracy: {round(score*100,2)}%')

logistic_model = LogisticRegression()
logistic_model.fit(x_vector_train, y_train)
# calculate accuracy
train_score = accuracy_score(y_train, logistic_model.predict(x_vector_train))
test_score = accuracy_score(y_test, logistic_model.predict(x_vector_test))
print(f'Accuracy for Logistics training model: {round(train_score*100,2)}%')
print(f'Accuracy for Logistics test model: {round(test_score*100,2)}%')

decision_model = DecisionTreeClassifier()
decision_model.fit(x_vector_train, y_train)
  
# testing the model
decision_train_score = accuracy_score(y_train, decision_model.predict(x_vector_train))
decision_test_score = accuracy_score(y_test, decision_model.predict(x_vector_test))
print(f'Accuracy for Decisions training model: {round(decision_train_score*100,2)}%')
print(f'Accuracy for Decisions test model: {round(decision_test_score*100,2)}%')

# Confusion matrix of results from Decision Tree classification 
conf_matrix = metrics.confusion_matrix(y_test, decision_model.predict(x_vector_test))
  
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                            display_labels=[False, True])
  
cm_display.plot()
plt.show()