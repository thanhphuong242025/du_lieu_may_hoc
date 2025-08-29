from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model (không cần scaler)
model = joblib.load("logistic_regression_model.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form (7 thuộc tính quan trọng)
            age = float(request.form["age"])
            sex = float(request.form["sex"])
            cp = float(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            thalach = float(request.form["thalach"])
            exang = float(request.form["exang"])

            # Gom dữ liệu thành numpy array trực tiếp
            sample = np.array([[age, sex, cp, trestbps, chol, thalach, exang]])

            # Dự đoán trực tiếp
            prediction = model.predict(sample)[0]
            result = " ⚠️ Phát hiện dấu hiệu tiềm ẩn về tim mạch. Hãy thăm khám bác sĩ để đảm bảo sức khỏe." if prediction == 1 else " 💚 Hiện tại tim mạch bạn ổn định. Hãy tiếp tục duy trì lối sống lành mạnh."

        except Exception as e:
            print("Lỗi khi xử lý dữ liệu:", e)
            result = "⚠️ Lỗi dữ liệu nhập, vui lòng thử lại!"

    return render_template("predict.html", result=result)

@app.route("/help")
def help_page():
    return render_template("help.html")

if __name__ == "__main__":
    app.run(debug=True)
