# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Encode target label
    le = LabelEncoder()
    df['Result'] = le.fit_transform(df['Result'])  # 'Pass' -> 1, 'Fail' -> 0

    # Features and target
    X = df.drop('Result', axis=1)
    y = df['Result']

    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
def train_and_evaluate():
    X_train, X_test, y_train, y_test = load_and_preprocess('../data/student_marks.csv')

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Save model
    joblib.dump(model, '../models/student_model.pkl')

if __name__ == "__main__":
    train_and_evaluate()
