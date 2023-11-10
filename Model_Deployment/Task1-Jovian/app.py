from flask import Flask, render_template, request
import pickle

cv = pickle.load(open("model/cv.pkl","rb"))
clf = pickle.load(open("model/clf.pkl","rb"))

# Create an Instance of the Flask Class
app = Flask(__name__)

# register a route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    email = request.form.get('email-content')
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template('index.html',prediction=prediction, email=email)


# running the application
if __name__=='__main__':
    app.run(host='0.0.0.0', port=8050, debug=True)