import pandas as pd
import joblib


# Load test data with specified encoding
test_df = pd.read_csv('../data/test.csv', encoding='ISO-8859-1')  # Use a different encoding


# Load the trained model and TF-IDF vectorizer
model = joblib.load('../models/logistic_model.pkl')
tfidf = joblib.load('../models/tfidf_vectorizer.pkl')  # Load the pre-trained TF-IDF vectorizer


# Transform the test data using the loaded vectorizer
X_test_tfidf = tfidf.transform(test_df['text'])  # Use transform instead of fit_transform


# Make predictions
test_pred = model.predict(X_test_tfidf)


# Prepare submission
submission = pd.DataFrame({'Index': test_df['Index'], 'target': test_pred})
submission.to_csv('../submission.csv', index=False)


print("Predictions saved to '../submission.csv'")
