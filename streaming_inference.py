import pandas as pd
import joblib
import time

model = joblib.load("model.pkl")
unseen = pd.read_csv("unseen_passengers.csv")

print("Starting streaming inference...")

for i in range(min(10, len(unseen))):
    passenger = unseen.iloc[[i]]
    prediction = int(model.predict(passenger)[0])
    result = "Survived" if prediction == 1 else "Did not survive"
    print(f"Passenger {i + 1}: {result}")
    time.sleep(1)

print("Streaming inference completed!")
