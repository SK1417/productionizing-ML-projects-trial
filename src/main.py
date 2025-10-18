from fastapi import FastAPI, File, UploadFile
from pydantic import Field, BaseModel
from sklearn.preprocessing import LabelEncoder
from typing import Literal
import joblib
import pandas as pd 

pipeline = joblib.load('pipeline.pkl')
label_encoder: LabelEncoder = joblib.load('label_encoder.pkl')

app = FastAPI(title="Telco Churn Predictor", version="1.0")

class TelcoCustomer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: float = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    return {"message": "✅ Telco Churn Model API is running!"}

@app.post("/predict")
async def predict(customer: TelcoCustomer) -> dict:
    df = pd.DataFrame([customer.dict()])
    pred = pipeline.predict(df)[0]
    churn_label = label_encoder.inverse_transform([pred])[0]
    return {"prediction": churn_label}