import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support


def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df[['job_type', 'category', 'skills']] = df[['job_type', 'category', 'skills']].fillna('Não Informado')

    df['job_type'] = df['job_type'].astype(str).str.strip().str.replace(r'[\s-]+', ' ', regex=True).str.lower()
    df['job_type'] = df['job_type'].replace({
        'não informado': 'Não Informado',
        'full time': 'Full-time',
        'fulltime': 'Full-time',
        'part time': 'Part-time',
        'parttime': 'Part-time',
        'remote': 'Remote',
        'contract': 'Contract',
        'manager': 'Manager'
    })

    df['salary_min'] = pd.to_numeric(df['salary_min'], errors='coerce').fillna(0.0)
    df['salary_max'] = pd.to_numeric(df['salary_max'], errors='coerce').fillna(0.0)
    df['salary_avg'] = (df['salary_min'] + df['salary_max']) / 2

    df['experience_required'] = pd.to_numeric(df['experience_required'], errors='coerce')
    df['experience_required'] = df.groupby('job_title')['experience_required'].transform(
        lambda x: x.fillna(x.median()) if not np.isnan(x.median()) else x
    )
    df = df.dropna(subset=['experience_required'])
    df['experience_required'] = df['experience_required'].astype(int)

    df['salary_class'] = pd.qcut(df['salary_avg'], q=3, labels=['Low', 'Medium', 'High'])
    df = df.dropna(subset=['salary_class'])
    df['salary_class'] = df['salary_class'].astype(str)

    return df


def compute_priors(df, target_col):
    counts = df[target_col].value_counts().sort_index()
    priors = counts / counts.sum()
    return priors


def compute_likelihoods(df, target_col, feature_cols, alpha=1.0):
    likelihoods = {}
    for feature in feature_cols:
        table = pd.crosstab(df[feature], df[target_col])
        smoothed = (table + alpha).div(table.sum(axis=0) + alpha * table.shape[0], axis=1)
        likelihoods[feature] = smoothed
    return likelihoods


def bayes_predict(df, test_df, target_col, feature_cols, alpha=1.0):
    priors = compute_priors(df, target_col)
    likelihoods = compute_likelihoods(df, target_col, feature_cols, alpha=alpha)
    classes = priors.index.tolist()

    def score_row(row):
        scores = {}
        for cls in classes:
            log_score = np.log(priors.loc[cls])
            for feature in feature_cols:
                value = row[feature]
                if value in likelihoods[feature].index:
                    prob = likelihoods[feature].loc[value, cls]
                else:
                    n_values = likelihoods[feature].shape[0]
                    prob = alpha / (likelihoods[feature].shape[0] + alpha * n_values)
                log_score += np.log(prob)
            scores[cls] = log_score
        return max(scores, key=scores.get)

    y_pred = test_df.apply(score_row, axis=1)
    return y_pred, priors, likelihoods


def encode_features(df, feature_cols):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X = encoder.fit_transform(df[feature_cols])
    return X, encoder


def encode_features_with(encoder, df, feature_cols):
    return encoder.transform(df[feature_cols])


def evaluate_model(y_true, y_pred, label_order=None):
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Classification report:')
    print(classification_report(y_true, y_pred, labels=label_order, zero_division=0))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred, labels=label_order))
    print()


def main():
    data_path = Path('data/job_market.csv')
    df = load_and_prepare_data(data_path)

    target_col = 'salary_class'
    feature_cols = ['job_title', 'job_type']

    print('Dataset shape:', df.shape)
    print('Target classes distribution:')
    print(df[target_col].value_counts(normalize=True).sort_index())
    print()

    priors = compute_priors(df, target_col)
    print('Priors (a priori) for salary_class:')
    print(priors)
    print()

    likelihoods = compute_likelihoods(df, target_col, feature_cols, alpha=1.0)
    for feature in feature_cols:
        print(f'Likelihoods P({feature}|salary_class):')
        print(likelihoods[feature].head(10))
        print()

    train_df, test_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df[target_col])

    print('Training shape:', train_df.shape)
    print('Test shape:', test_df.shape)
    print()

    y_test = test_df[target_col]
    y_pred_bayes, priors_train, likelihoods_train = bayes_predict(train_df, test_df, target_col, feature_cols, alpha=1.0)

    print('=== Bayes classifier (manual) ===')
    evaluate_model(y_test, y_pred_bayes, label_order=['Low', 'Medium', 'High'])

    X_train, encoder = encode_features(train_df, feature_cols)
    X_test = encode_features_with(encoder, test_df, feature_cols)
    y_train = train_df[target_col]

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print('=== Decision Tree ===')
    evaluate_model(y_test, y_pred_dt, label_order=['Low', 'Medium', 'High'])

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print('=== Logistic Regression ===')
    evaluate_model(y_test, y_pred_lr, label_order=['Low', 'Medium', 'High'])

    print('Comparison:')
    print('Bayes accuracy:', accuracy_score(y_test, y_pred_bayes))
    print('Decision Tree accuracy:', accuracy_score(y_test, y_pred_dt))
    print('Logistic Regression accuracy:', accuracy_score(y_test, y_pred_lr))

    example = pd.DataFrame([
        {'job_title': 'Software Engineer', 'job_type': 'Full-time'},
        {'job_title': 'Machine Learning Engineer', 'job_type': 'Remote'},
        {'job_title': 'Product Manager', 'job_type': 'Contract'},
    ])
    print('\nExample posterior probabilities for manual Bayes:')
    for _, row in example.iterrows():
        scores = {}
        for cls in priors_train.index:
            log_score = np.log(priors_train.loc[cls])
            for feature in feature_cols:
                value = row[feature]
                if value in likelihoods_train[feature].index:
                    prob = likelihoods_train[feature].loc[value, cls]
                else:
                    n_values = likelihoods_train[feature].shape[0]
                    prob = 1.0 / (likelihoods_train[feature].shape[0] + n_values)
                log_score += np.log(prob)
            scores[cls] = log_score
        exp_scores = {cls: np.exp(score - np.max(list(scores.values()))) for cls, score in scores.items()}
        probs = {cls: exp_scores[cls] / sum(exp_scores.values()) for cls in scores}
        print(row.to_dict(), probs)


if __name__ == '__main__':
    main()
