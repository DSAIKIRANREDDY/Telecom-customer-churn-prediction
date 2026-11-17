import pandas as pd
import re

from pathlib import Path
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import UploadFile, File, HTTPException
from fastapi import FastAPI, Request
from pydantic import ValidationError
from pandas.errors import EmptyDataError, ParserError

from telecom_churn.inference.model_loader import ModelLoader
from telecom_churn.inference.predictor import ChurnPredictor
from telecom_churn.inference.explainability import CustomerExplainability
from telecom_churn.utils.logger import get_logger
from fastapi import FastAPI, Request
from telecom_churn.api.schemas import (CustomerRequest, PredictionResponse)

logger = get_logger(__name__)

MODEL_PATH = Path(
    "artifacts/models/tuned/xgboost_tuned_model.pkl"
)
EXPLANATION_DIR = Path(
    "artifacts/reports/explainability/inference"
)
MAX_BATCH_SIZE = 5000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

def validate_upload_size(file: UploadFile) -> None:

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the maximum allowed size of 10 MB"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Telecom Churn API")

    # Load model once during application startup
    model_loader = ModelLoader(MODEL_PATH)

    model_loader.load_model()
    model_loader.load_explainer()

    # Initialize inference services
    predictor = ChurnPredictor(
        model_loader=model_loader,
        threshold=0.65
    )

    customer_explainer = CustomerExplainability(
        model_loader=model_loader,
        output_dir=EXPLANATION_DIR
    )

    # Make services available to API endpoints
    app.state.model_loader = model_loader
    app.state.predictor = predictor
    app.state.customer_explainer = customer_explainer

    logger.info("Telecom Churn API startup completed")

    yield

    logger.info("Shutting down Telecom Churn API")


app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="API for customer churn prediction and explainability",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():

    return {
        "message": "Telecom Customer Churn Prediction API",
        "status": "running"
    }



@app.get("/health")
def health():
    # Check whether the model is loaded
    model_loaded = (
        hasattr(app.state, "model_loader")
        and app.state.model_loader.model is not None
    )

    # Check whether the SHAP explainer is initialized
    explainer_loaded = (
        hasattr(app.state, "model_loader")
        and app.state.model_loader.explainer is not None
    )

    # Check whether the predictor is initialized
    predictor_loaded = (
        hasattr(app.state, "predictor")
        and app.state.predictor is not None
    )

    # Determine overall application health
    is_healthy = (
        model_loaded
        and explainer_loaded
        and predictor_loaded
    )

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "components": {
            "model_loaded": model_loaded,
            "explainer_loaded": explainer_loaded,
            "predictor_loaded": predictor_loaded
        }
    }


@app.get("/health/live")
def health_live():

    return {
        "status": "alive"
    }



@app.get("/health/ready")
def health_ready():

    model_loaded = (
        hasattr(app.state, "model_loader")
        and app.state.model_loader.model is not None
    )

    explainer_loaded = (
        hasattr(app.state, "model_loader")
        and app.state.model_loader.explainer is not None
    )

    predictor_loaded = (
        hasattr(app.state, "predictor")
        and app.state.predictor is not None
    )

    is_ready = (
        model_loaded
        and explainer_loaded
        and predictor_loaded
    )

    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "components": {
                    "model_loaded": model_loaded,
                    "explainer_loaded": explainer_loaded,
                    "predictor_loaded": predictor_loaded
                }
            }
        )

    return {
        "status": "ready",
        "components": {
            "model_loaded": model_loaded,
            "explainer_loaded": explainer_loaded,
            "predictor_loaded": predictor_loaded
        }
    }



@app.post( "/predict", response_model=PredictionResponse)
def predict_customer( customer: CustomerRequest, request: Request):
    logger.info("Received customer prediction request")

    # Get the already initialized predictor
    predictor = request.app.state.predictor

    # Convert validated Pydantic data into a dictionary
    customer_data = customer.model_dump(
        exclude_none=True
    )

    # Generate prediction
    result = predictor.predict_single(customer_data)

    logger.info(
        "Customer prediction completed: %s",
        result.get("customerID", "unknown")
    )

    return result




@app.post("/predict/batch")
def predict_batch(request: Request, file: UploadFile = File(...)):

    logger.info("Received batch prediction request: %s", file.filename)

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file"
        )
    validate_upload_size(file)
    try:
        # Read uploaded CSV
        try:
            customers_df = pd.read_csv(file.file)
        except EmptyDataError:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV is empty"
            )
        except (ParserError, UnicodeDecodeError):
            raise HTTPException(
                status_code=400,
                detail="Unable to read CSV. Please check the file format."
            )

        if customers_df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV contains no customer records"
            )

        if len(customers_df) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size exceeds the maximum limit of {MAX_BATCH_SIZE} customers"
            )

        logger.info("Batch input loaded. Shape: %s", customers_df.shape)

        predictor = request.app.state.predictor

        # Validate CSV columns
        expected_columns = set(predictor.expected_features)
        allowed_columns = expected_columns | {"customerID"}

        missing_columns = expected_columns - set(customers_df.columns)
        unexpected_columns = set(customers_df.columns) - allowed_columns

        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required columns: {sorted(missing_columns)}"
            )

        if unexpected_columns:
            raise HTTPException(
                status_code=422,
                detail=f"Unexpected columns: {sorted(unexpected_columns)}"
            )

        # Validate each customer record using Pydantic
        validation_errors = []

        for index, row in customers_df.iterrows():
            try:
                CustomerRequest.model_validate(row.to_dict())
            except ValidationError as error:
                for item in error.errors():
                    validation_errors.append({
                        "row": int(index) + 2,
                        "field": ".".join(str(x) for x in item["loc"]),
                        "message": item["msg"]
                    })

        if validation_errors:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "CSV contains invalid customer data",
                    "errors": validation_errors
                }
            )

        # Generate predictions
        results = predictor.predict_batch(customers_df)

        predictions = results.to_dict(orient="records")

        logger.info(
            "Batch prediction completed for %s customers",
            len(predictions)
        )

        return {
            "total_customers": len(predictions),
            "predictions": predictions
        }

    except HTTPException:
        raise

    except (ValueError, TypeError) as error:
        logger.exception("Invalid batch prediction input")
        raise HTTPException(status_code=400, detail=str(error))

    except Exception:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )

    finally:
        file.file.close()




@app.post("/explain")
def explain_customer( customer: CustomerRequest, request: Request):

    logger.info("Received customer explanation request")

    customer_explainer = request.app.state.customer_explainer

    customer_data = customer.model_dump(
        exclude_none=True
    )

    try:

        # Generate customer-level SHAP explanation
        result = customer_explainer.explain_customer(
            customer_data
        )

        # Return original-feature contributions
        contributions = result["contributions"].to_dict(
            orient="records"
        )

        customer_id = result["customerID"]

        logger.info(
            "Customer explanation completed: %s",
            customer_id
        )

        return {
            "customerID": customer_id,
            "base_value": result["base_value"],
            "contributions": contributions,
            "plot_url": f"/explain/plot/{customer_id}"
        }

    except (ValueError, TypeError) as error:

        logger.exception("Invalid explanation request")

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception("Customer explanation failed")

        raise HTTPException(
            status_code=500,
            detail="Customer explanation failed"
        )



@app.get("/explain/plot/{customer_id}")
def get_customer_explanation_plot(customer_id: str):

    # Apply the same filename sanitization used by SHAP
    safe_customer_id = re.sub(
        r"[^A-Za-z0-9_-]",
        "_",
        customer_id
    )

    plot_path = (
        EXPLANATION_DIR /
        f"customer_shap_{safe_customer_id}.png"
    )

    if not plot_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Customer explanation plot not found"
        )

    return FileResponse(
        path=plot_path,
        media_type="image/png",
        filename=f"customer_shap_{safe_customer_id}.png"
    )
