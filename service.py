from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
from starlette.responses import StreamingResponse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from train_and_save_model import EncoderData
import io

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def item_to_dataframe(item: Item):
    return pd.DataFrame([item.dict()])


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    x = item_to_dataframe(item)
    p = model.predict(x)[0]
    return p


@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> StreamingResponse:

    df = pd.read_csv(io.StringIO(str(file.file.read(), 'utf-8')))
    df1 = df.copy(deep=True)

    df1['selling_price'] = model.predict(df)

    output = io.StringIO()
    df1.to_csv(output, index=False)

    output.seek(0)
    return StreamingResponse(io.BytesIO(output.getvalue().encode()), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=processed.csv"})


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

