from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        target_column = request.form["target"]

        df = pd.read_csv(file)

        # Encode categorical columns
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = LabelEncoder().fit_transform(df[col])

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        }

        best_model = None
        best_score = 0

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = accuracy_score(y_test, preds)

            if score > best_score:
                best_score = score
                best_model = name

        result = f"Best Model: {best_model} | Accuracy: {best_score:.2f}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)