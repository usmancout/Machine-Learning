from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('../Code/Random_Forest.pkl')
scaler=joblib.load('../Code/scaler_model.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = float(request.form.get('Pclass'))
        sex = float(request.form.get('Sex'))
        age = float(request.form.get('Age'))
        sibsp = float(request.form.get('SibSp'))
        parch = float(request.form.get('Parch'))
        fare = float(request.form.get('Fare'))
        embarked = float(request.form.get('Embarked'))

        user_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

        user_data_scaled=scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)

        result_text = "Survived" if prediction[0] == 1 else "Did Not Survive"
        return render_template('result.html', prediction=result_text)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
