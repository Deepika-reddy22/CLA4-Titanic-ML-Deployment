# CLA 4: ML Model Deployment Strategies and Model Serving using FastAPI

## Project Overview
This project demonstrates machine learning model training, online inference, batch inference, streaming inference, and model serving using FastAPI.

## Dataset
**Titanic - Machine Learning from Disaster**

Dataset file used: `train.csv`

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn

## Project Files
- `train_model.py` – trains and evaluates the machine learning model.
- `model.pkl` – saved trained model pipeline.
- `online_inference.py` – predicts one passenger's survival.
- `batch_inference.py` – predicts survival for multiple passengers.
- `streaming_inference.py` – simulates streaming predictions with a one-second delay.
- `batch_predictions.csv` – stores batch prediction results.
- `app.py` – implements the FastAPI application.
- `client.py` – sends prediction requests to the API.
- `requirements.txt` – lists the required Python packages.
- `observations_report.pdf` – contains the assignment report.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

### 3. Run online inference
```bash
python online_inference.py
```

### 4. Run batch inference
```bash
python batch_inference.py
```

### 5. Run streaming inference
```bash
python streaming_inference.py
```

### 6. Start the FastAPI server
```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API.

## API Endpoints
- `GET /health` – checks whether the API is running.
- `POST /predict` – predicts passenger survival.

## Observations
1. Online inference predicts one record immediately.
2. Batch inference processes multiple records together.
3. Streaming inference processes records one by one as they arrive.
4. FastAPI provides endpoints and automatic API documentation.
5. Reusing the same saved model provides consistent predictions.

## Conclusion
The project demonstrates a complete machine learning workflow, from model training and saving to different inference strategies and REST API deployment.
