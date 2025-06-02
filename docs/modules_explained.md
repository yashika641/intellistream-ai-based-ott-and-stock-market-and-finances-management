````markdown
# IntelliStreamAI Module API Reference

## Overview
This document provides the API endpoints for each core module of the IntelliStreamAI platform. The APIs are designed for model training, inference, and related operations, enabling integration with front-end applications and other services.

---

## 1. OTT User Behavior Analysis API

### Endpoints

- **POST /api/ott/user-behavior/train**  
  Train the user behavior prediction model with historical user data.

  **Request Body:**  
  ```json
  {
    "user_activity_data": "CSV or JSON formatted data",
    "demographics_data": "CSV or JSON formatted data"
  }
````

**Response:**

```json
{
  "status": "success",
  "message": "User behavior model trained successfully"
}
```

* **POST /api/ott/user-behavior/predict**
  Predict user engagement or churn likelihood.

  **Request Body:**

  ```json
  {
    "user_id": "string",
    "session_features": { /* feature key-value pairs */ }
  }
  ```

  **Response:**

  ```json
  {
    "user_id": "string",
    "engagement_score": 0.85,
    "churn_risk": 0.12
  }
  ```

---

## 2. Content Recommendation System API

### Endpoints

* **POST /api/recommendation/train**
  Train recommendation model with user-item interactions.

  **Request Body:**

  ```json
  {
    "interaction_data": "CSV or JSON formatted data",
    "content_metadata": "CSV or JSON formatted data"
  }
  ```

  **Response:**

  ```json
  {
    "status": "success",
    "message": "Recommendation model trained successfully"
  }
  ```

* **POST /api/recommendation/get**
  Get content recommendations for a user.

  **Request Body:**

  ```json
  {
    "user_id": "string",
    "top_n": 10
  }
  ```

  **Response:**

  ```json
  {
    "user_id": "string",
    "recommendations": [
      { "content_id": "string", "score": 0.95 },
      { "content_id": "string", "score": 0.91 }
    ]
  }
  ```

---

## 3. Script Success Prediction API

### Endpoints

* **POST /api/script-success/train**
  Train the script success prediction model using historical script data.

  **Request Body:**

  ```json
  {
    "script_metadata": "CSV or JSON formatted data",
    "script_text_features": "Extracted NLP features"
  }
  ```

  **Response:**

  ```json
  {
    "status": "success",
    "message": "Script success prediction model trained"
  }
  ```

* **POST /api/script-success/predict**
  Predict success score for a new script.

  **Request Body:**

  ```json
  {
    "script_metadata": { /* metadata key-values */ },
    "script_text_features": { /* NLP feature values */ }
  }
  ```

  **Response:**

  ```json
  {
    "predicted_success_score": 0.78,
    "key_influencers": {
      "genre": 0.3,
      "sentiment": 0.25,
      "cast_popularity": 0.2
    }
  }
  ```

---

## 4. Financial Market Correlation API

### Endpoints

* **POST /api/finance-correlation/train**
  Train correlation models with OTT and financial time-series data.

  **Request Body:**

  ```json
  {
    "ott_time_series": "Time-series data",
    "financial_time_series": "Time-series data"
  }
  ```

  **Response:**

  ```json
  {
    "status": "success",
    "message": "Financial correlation model trained"
  }
  ```

* **POST /api/finance-correlation/predict**
  Predict financial market movement based on OTT trends.

  **Request Body:**

  ```json
  {
    "ott_recent_data": "Recent OTT consumption metrics"
  }
  ```

  **Response:**

  ```json
  {
    "predicted_market_trend": "bullish",
    "confidence_score": 0.87
  }
  ```

---

## 5. Data Pipeline API

### Endpoints

* **POST /api/data/ingest**
  Ingest raw data streams into the system.

  **Request Body:**

  ```json
  {
    "data_source": "string",
    "data_payload": "raw data"
  }
  ```

  **Response:**

  ```json
  {
    "status": "success",
    "message": "Data ingested successfully"
  }
  ```

* **GET /api/data/status**
  Get status of data ingestion and preprocessing jobs.

  **Response:**

  ```json
  {
    "status": "completed",
    "last_ingested_timestamp": "ISO 8601 datetime"
  }
  ```

---

## 6. Model Explainability API

### Endpoints

* **POST /api/explainability/generate**
  Generate model explanation reports.

  **Request Body:**

  ```json
  {
    "model_name": "string",
    "input_data": "feature set"
  }
  ```

  **Response:**

  ```json
  {
    "explanation": "SHAP/LIME values",
    "visualization_url": "http://..."
  }
  ```

---

## 7. Deployment & Inference API

### Endpoints

* **POST /api/predict**
  General prediction endpoint for deployed models.

  **Request Body:**

  ```json
  {
    "model_name": "string",
    "input_features": { /* feature key-values */ }
  }
  ```

  **Response:**

  ```json
  {
    "prediction": "model output"
  }
  ```

---

 
