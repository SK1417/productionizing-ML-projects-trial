import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


FILE_NAME = 'data/churn.csv'
df = pd.read_csv(FILE_NAME)
num_cols = ['TotalCharges', 'MonthlyCharges', 'tenure']
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
df = df.drop(columns=['customerID'], axis=1)

cols_le = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
cols_ohe = [col for col in df.columns if col not in cols_le + num_cols + ['Churn']]

preprocessor = ColumnTransformer(
    transformers=[
        ('labels', OrdinalEncoder(), cols_le),
        ('one_hot_encoding', OneHotEncoder(sparse_output=False), cols_ohe),
        ('scaling', StandardScaler(), num_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier())
])


X = df.drop(columns=['Churn'])
y = df['Churn'].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

pipeline.fit(X_train, y_train)

print("Train Accuracy:", pipeline.score(X_train, y_train))
print("Test Accuracy:", pipeline.score(X_test, y_test))

joblib.dump(pipeline, 'pipeline.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')