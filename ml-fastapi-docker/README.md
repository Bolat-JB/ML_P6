# Adult Census Income Prediction API (FastAPI + Docker)

A machine learning service that predicts whether an individual's annual income exceeds `$50K` (`<=50K` vs `>50K`) based on demographic and employment attributes, served through FastAPI and packaged with Docker.

## Project Structure

```text
ml-fastapi-docker/
├── train.py           # Loads dataset, fits sklearn Pipeline, prints metrics, saves model.joblib
├── model.joblib       # Trained preprocessing + HistGradientBoostingClassifier pipeline artifact
├── main.py            # FastAPI application exposing GET / and POST /predict
├── requirements.txt   # Pinned Python dependencies
├── Dockerfile         # Container build instructions (python:3.12-slim)
└── README.md          # Project overview and usage instructions
```

## Dataset & Model Pipeline

- **Dataset**: Adult Census Income loaded from OpenML via `fetch_openml("adult", version=2, as_frame=True)` (48,842 records).
- **Preprocessing**:
  - Column hyphens normalized to underscores (`marital-status` -> `marital_status`, etc.).
  - `fnlwgt` (sampling weight unavailable at inference) and `education_num` (redundant with `education`) removed, leaving **12 input features**:
    `age`, `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `capital_gain`, `capital_loss`, `hours_per_week`, `native_country`.
  - Missing values (`'?'`) converted to `np.nan`; target mapped (`<=50K` -> `0`, `>50K` -> `1`).
  - `ColumnTransformer` inside a single `scikit-learn` `Pipeline`:
    - Numeric (`age`, `capital_gain`, `capital_loss`, `hours_per_week`): `SimpleImputer(strategy="median")` + `StandardScaler()`
    - Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`
- **Classifier**: `HistGradientBoostingClassifier(max_depth=8, random_state=42)` trained on a stratified 80/20 split.
- **Test Metrics**:
  - Accuracy: `0.8765`
  - Precision: `0.7998`
  - Recall: `0.6459`
  - F1-score: `0.7146`
  - ROC-AUC: `0.9289`

## Quick Start (Local)

1. Navigate to the project folder and install dependencies:
   ```bash
   cd ml-fastapi-docker
   pip install -r requirements.txt
   ```

2. Train the model and generate `model.joblib`:
   ```bash
   python train.py
   ```

3. Start the FastAPI server with Uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. Open interactive Swagger UI in your browser:
   - `http://localhost:8000/docs`

## Run with Docker

1. Build the Docker image:
   ```bash
   docker build -t ml-fastapi-adult .
   ```

2. Run the container on port `8000`:
   ```bash
   docker run -d --name ml-fastapi-adult-container -p 8000:8000 ml-fastapi-adult
   ```

3. Check logs or stop the container when finished:
   ```bash
   docker logs ml-fastapi-adult-container
   docker stop ml-fastapi-adult-container && docker rm ml-fastapi-adult-container
   ```

## API Endpoints & Example Requests

### Health Check (`GET /`)
```bash
curl http://localhost:8000/
```
Response:
```json
{"message": "ML API is running"}
```

### Income Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 39,
    "workclass": "Private",
    "education": "Bachelors",
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 5178,
    "capital_loss": 0,
    "hours_per_week": 45,
    "native_country": "United-States"
  }'
```

Example response:
```json
{
  "prediction": 1,
  "income_class": ">50K",
  "probability_over_50k": 0.9741
}
```
