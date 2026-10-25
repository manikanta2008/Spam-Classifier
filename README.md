# Email Spam Classifier

NLP project that classifies messages as **spam** or **ham** (not spam) using TF-IDF features and a Multinomial Naive Bayes model, built with Python and Scikit-learn.

## Overview

- **Goal:** automatically detect spam emails and text messages
- **Type:** supervised learning, binary text classification
- **Dataset:** [dataset name and source, e.g. SMS Spam Collection or Kaggle link], [N] messages, [N]% spam
- **Target:** [label column, e.g. spam = 1, ham = 0]

## Tech Stack

Python, Scikit-learn, NLTK, Pandas, NumPy, Matplotlib, Jupyter Notebook

## Workflow

1. **Data cleaning:** removed duplicates and null values, encoded labels
2. **Text preprocessing:** lowercasing, tokenization, stop-word removal and stemming with NLTK
3. **Feature extraction:** converted cleaned text to numerical vectors using TF-IDF
4. **Model training:** trained a Multinomial Naive Bayes classifier on a [80/20] train-test split
5. **Evaluation:** measured accuracy, precision, recall and F1-score, and plotted a confusion matrix

## Results

| Metric | Score |
|---|---|
| Accuracy | [ ] |
| Precision | [ ] |
| Recall | [ ] |
| F1-score | [ ] |

[One line on the result, e.g. which class was harder to predict, or how the model compared with other classifiers you tried.]

## Project Structure

```
email-spam-classifier/
├── data/
│   └── spam.csv
├── notebooks/
│   └── spam_classifier.ipynb
├── requirements.txt
└── README.md
```

## How to Run

```bash
git clone https://github.com/manikanta2008/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
jupyter notebook notebooks/spam_classifier.ipynb
```

## Example Prediction

```python
message = ["Congratulations! You have won a free prize. Click here to claim."]
features = vectorizer.transform(message)
print(model.predict(features))
```

## Future Improvements

- Compare with Logistic Regression, SVM and Random Forest
- Try n-gram TF-IDF features and hyperparameter tuning
- Serve the model through a FastAPI endpoint with a simple web form

## Author

**Manikanta Repaka**
[GitHub](https://github.com/manikanta2008) · [LinkedIn](https://www.linkedin.com/in/mani-kanta-repaka/)
