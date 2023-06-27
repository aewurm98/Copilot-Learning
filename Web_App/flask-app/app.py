import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

app = Flask(__name__)

# Challenge: rather than loading the externally-trained model and making the prediction in the flask app
# Train the model in the app itself via a different route
# model = pickle.load(open("../ufos_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )

@app.route("/train", methods=["POST"])
def train():
    
    df = pd.read_csv('../ufos_clean.csv')

    # Select Seconds, Latitude, and Longitude as the features (X) for the model and Country as the label (y)
    X = df[['Seconds', 'Latitude', 'Longitude']]
    y = df['Country']    

    tst_params = [int(x) for x in request.form.values()]
    final_params = [np.array(tst_params)]

    # Split the data into training and test sets with 20% of the data in the test set and a random_state of 0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tst_params[0]/100, random_state=tst_params[1])

    # Create a logistic regression model named model and fit it to the training data
    global model
    model = LogisticRegression().fit(X_train, y_train)

    # Declare a variable for the model's predictions on the test data
    y_pred = model.predict(X_test)

    return render_template(
        "index.html", confirmation_msg="Model trained successfully!"
    )


if __name__ == "__main__":
    app.run(debug=True)