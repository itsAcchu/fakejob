import os
from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

app = Flask(__name__)

# Initialize the model and vectorizer
model = None
tfidf = None

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def load_model():
    global model, tfidf
    model = RandomForestClassifier(random_state=42)
    tfidf = TfidfVectorizer(max_features=5000)

def process_data(file_path):
    data = pd.read_csv(file_path)

    text_columns = ['title', 'description', 'requirements', 'benefits']
    data['combined_text'] = data[text_columns].fillna('').agg(' '.join, axis=1)

    y = data['fraudulent']
    X = tfidf.fit_transform(data['combined_text'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    
    total_fake_jobs = sum(y_test)
    predicted_fake_jobs = sum(y_pred)
    correctly_predicted_fake_jobs = sum((y_test == 1) & (y_pred == 1))

    return report, total_fake_jobs, predicted_fake_jobs, correctly_predicted_fake_jobs


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            report, total_fake_jobs, predicted_fake_jobs, correctly_predicted_fake_jobs = process_data(file_path)
            return render_template('report.html', report=report,
                                   total_fake_jobs=total_fake_jobs,
                                   predicted_fake_jobs=predicted_fake_jobs,
                                   correctly_predicted_fake_jobs=correctly_predicted_fake_jobs)
    return render_template('upload.html')

if __name__ == '__main__':
    load_model()
    app.run(debug=True)