import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create synthetic movie reviews
reviews = [
    "This movie was absolutely fantastic and amazing",
    "I loved every minute of this film",
    "Best movie I have seen in years",
    "Wonderful performances and great story",
    "Highly recommend this beautiful film",
    "Outstanding cinematography and acting",
    "A masterpiece of modern cinema",
    "Brilliant direction and superb cast",
    "This film moved me deeply and was inspiring",
    "Incredible movie with a powerful message",
    "This movie was terrible and boring",
    "I hated every minute of this film",
    "Worst movie I have seen in years",
    "Awful performances and poor story",
    "Do not waste your time on this film",
    "Disappointing and poorly directed movie",
    "A complete waste of money and time",
    "Terrible acting and no plot whatsoever",
    "This film was dull and completely pointless",
    "Incredibly boring with no redeeming qualities",
]

# 1 means positive, 0 means negative
labels = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]

df = pd.DataFrame({'review': reviews, 'sentiment': labels})

print(df.head(10))
print()
print(f"Total reviews: {len(df)}")
print(f"Positive: {sum(labels)}")
print(f"Negative: {len(labels) - sum(labels)}")

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df['review'])

print("Shape of transformed data:")
print(x.shape)
print()
print("Sample of the vocabulary it learned:")
print(vectorizer.get_feature_names_out()[:20])

x_train, x_test, y_train, y_test = train_test_split(
    x, df['sentiment'], test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print()
print("Flassification Report:")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive'])) 

new_reviews = [
    "This was an absolutely wonderful experience",
    "I completely hate this boring waste of time",
    "Fantastic performances and brilliant story",
    "Terrible and awful in every possible way",
]

new_vectors = vectorizer.transform(new_reviews)
predictions = model.predict(new_vectors)

for review, prediction in zip(new_reviews, predictions):
    sentiment = "POSTIVE" if prediction == 1 else "NEGATIVE"
    print(f"{sentiment}: {review}")



















































































































































































































































































