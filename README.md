# 🎓 Student Performance Predictor

This project predicts student performance (`Pass` or `Fail`) based on subject scores in Maths, Physics, and Chemistry using a machine learning model.

## 📁 Dataset

The dataset used is from Kaggle:  
[Student Marks Dataset](https://www.kaggle.com/datasets/vicky1999/student-marks-dataset)  
Columns: `maths`, `physics`, `chemistry`, `result`

---

## 🧰 Tech Stack

- Python 3.10+
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib

---

## 🛠️ Project Structure

Student-Performance-Predictor/
├── data/
│ └── student_marks.csv
├── src/
│ ├── preprocess.py
│ ├── train_model.py
│ └── model.joblib
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## ✅ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Student-Performance-Predictor.git
cd Student-Performance-Predictor
2. Create and activate a virtual environment
bash
Copy
Edit
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Add dataset
Download student_marks.csv from Kaggle and place it inside the data/ folder.

🚀 Run the Model
Navigate to the src folder:

bash
Copy
Edit
cd src
Run the training script:

bash
Copy
Edit
python train_model.py
You will see output like this:

yaml
Copy
Edit
Accuracy: 1.0
Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00       145
           1       1.00      1.00      1.00        55
    accuracy                           1.00       200
A file model.joblib will be saved which contains the trained model.

📊 Sample Output - Confusion Matrix
A confusion matrix plot will be shown after training the model, indicating performance.

