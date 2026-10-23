import pandas as pd
import joblib

model = joblib.load("model.pkl")
unseen = pd.read_csv("unseen_passengers.csv")

passenger = unseen.iloc[[0]]
prediction = int(model.predict(passenger)[0])

print("Passenger details:")
print(passenger)
print("Prediction:", "Survived" if prediction == 1 else "Did not survive")
