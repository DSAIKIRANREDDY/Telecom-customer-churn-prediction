
import re
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    HTTPException,
    Security
)
from fastapi.responses import FileResponse
from pydantic import ValidationError
from pandas.errors import EmptyDataError, ParserError

from telecom_churn.api.security import verify_api_key
from telecom_churn.inference.model_loader import ModelLoader
from telecom_churn.inference.predictor import ChurnPredictor
from telecom_churn.inference.explainability import CustomerExplainability
from telecom_churn.utils.logger import get_logger
from telecom_churn.api.schemas import (
    CustomerRequest,
    PredictionResponse
)


logger = get_logger(__name__)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(
    "artifacts/models/tuned/xgboost_tuned_model.pkl"
)

EXPLANATION_DIR = Path("artifacts/reports/explainability/inference")

MAX_BATCH_SIZE = 5000
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------

def validate_upload_size(file: UploadFile) -> None:

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Uploaded file exceeds the maximum allowed size of 10 MB"
        )


def load_and_validate_batch_csv( file: UploadFile, predictor: ChurnPredictor) -> pd.DataFrame:

    """
    Read and validate an uploaded customer CSV.

    Validates:
    1. File extension
    2. File size
    3. CSV readability
    4. Empty records
    5. Maximum batch size
    6. Required and unexpected columns
    7. Individual customer records using Pydantic
    """

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400,detail="Please upload a CSV file")

    validate_upload_size(file)

    # Read CSV
    try:

        customers_df = pd.read_csv(file.file)

    except EmptyDataError:

        raise HTTPException(status_code=400,detail="Uploaded CSV is empty")

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

    if customers_df.columns.duplicated().any():
        raise HTTPException(
            status_code=422,
            detail="CSV contains duplicate column names"
        )

    # Validate batch size
    if len(customers_df) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Batch size exceeds the maximum limit "
                f"of {MAX_BATCH_SIZE} customers"
            )
        )

    logger.info(
        "CSV loaded successfully. Shape: %s",
        customers_df.shape
    )

    # Validate columns
    expected_columns = set(predictor.expected_features)

    allowed_columns = expected_columns | {"customerID"}

    missing_columns = (
        expected_columns - set(customers_df.columns)
    )

    unexpected_columns = (
        set(customers_df.columns) - allowed_columns
    )

    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Required customer columns are missing",
                "missing_columns": sorted(missing_columns)
            }
        )

    if unexpected_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unexpected customer columns found",
                "unexpected_columns": sorted(unexpected_columns)
            }
        )

    # Validate individual records
    validation_errors = []

    for index, row in customers_df.iterrows():

        try:

            CustomerRequest.model_validate(row.to_dict())

        except ValidationError as error:

            for item in error.errors():

                validation_errors.append({
                    "row": int(index) + 2,
                    "field": ".".join(
                        str(x) for x in item["loc"]
                    ),
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

    return customers_df


# --------------------------------------------------
# APPLICATION LIFESPAN
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting Telecom Churn API")

    # Load model once during application startup
    model_loader = ModelLoader(MODEL_PATH)

    model_loader.load_model()
    model_loader.load_explainer()

    # Initialize prediction service
    predictor = ChurnPredictor(
        model_loader=model_loader,
        threshold=0.65
    )

    # Initialize explainability service
    customer_explainer = CustomerExplainability(
        model_loader=model_loader,
        output_dir=EXPLANATION_DIR
    )

    # Store services in application state
    app.state.model_loader = model_loader
    app.state.predictor = predictor
    app.state.customer_explainer = customer_explainer

    logger.info("Telecom Churn API startup completed")

    yield

    logger.info("Shutting down Telecom Churn API")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Telecom Customer Churn Prediction API",
    description="API for customer churn prediction and explainability",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Telecom Customer Churn Prediction API",
        "status": "running"
    }


# --------------------------------------------------
# HEALTH CHECK ENDPOINTS
# --------------------------------------------------

@app.get("/health")
def health(_: str = Security(verify_api_key)):

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
def health_ready(_: str = Security(verify_api_key)):

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


# --------------------------------------------------
# SINGLE CUSTOMER PREDICTION
# --------------------------------------------------

@app.post( "/predict", response_model=PredictionResponse)
def predict_customer( customer: CustomerRequest, request: Request,_: str = Security(verify_api_key)):

    logger.info("Received customer prediction request")

    predictor = request.app.state.predictor

    customer_data = customer.model_dump(
        exclude_none=True
    )

    result = predictor.predict_single(customer_data)

    logger.info("Customer prediction completed: %s",result.get("customerID", "unknown"))

    return result


# --------------------------------------------------
# BATCH CUSTOMER PREDICTION
# --------------------------------------------------

@app.post("/predict/batch")
def predict_batch(request: Request,file: UploadFile = File(...),_: str = Security(verify_api_key)):

    logger.info("Received batch prediction request: %s",file.filename)

    try:

        predictor = request.app.state.predictor

        customers_df = load_and_validate_batch_csv(file,predictor)

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

        logger.exception(
            "Invalid batch prediction input"
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception(
            "Batch prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Batch prediction failed"
        )

    finally:
        file.file.close()


# --------------------------------------------------
# INDIVIDUAL CUSTOMER SHAP EXPLAINABILITY
# --------------------------------------------------

@app.post("/explain")
def explain_customer(customer: CustomerRequest,request: Request,_: str = Security(verify_api_key)):

    logger.info("Received customer explanation request")

    customer_explainer = request.app.state.customer_explainer

    customer_data = customer.model_dump(exclude_none=True)

    try:

        result = customer_explainer.explain_customer(customer_data)

        contributions = result["contributions"].to_dict(orient="records")

        customer_id = result["customerID"]

        logger.info("Customer explanation completed: %s",customer_id)

        return {
            "customerID": customer_id,
            "base_value": result["base_value"],
            "contributions": contributions,
            "plot_url": f"/explain/plot/{customer_id}"
        }

    except (ValueError, TypeError) as error:

        logger.exception(
            "Invalid explanation request"
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception(
            "Customer explanation failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Customer explanation failed"
        )


# --------------------------------------------------
# INDIVIDUAL CUSTOMER WATERFALL PLOT
# --------------------------------------------------

@app.get("/explain/plot/{customer_id}")
def get_customer_explanation_plot(customer_id: str,_: str = Security(verify_api_key)):

    safe_customer_id = re.sub( r"[^A-Za-z0-9_-]","_",customer_id)

    plot_path = (EXPLANATION_DIR / f"customer_shap_{safe_customer_id}.png")

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


# --------------------------------------------------
# BATCH SHAP EXPLAINABILITY
# --------------------------------------------------

@app.post("/explain/batch")
def explain_batch(request: Request,file: UploadFile = File(...),_: str = Security(verify_api_key)):

    logger.info("Received batch explanation request: %s",file.filename)

    try:

        predictor = request.app.state.predictor

        # Reuse common CSV validation
        customers_df = load_and_validate_batch_csv(file,predictor)

        customer_explainer = (request.app.state.customer_explainer)

        # Generate batch SHAP explanations
        result = customer_explainer.explain_batch(customers_df)

        feature_importance = (result["feature_importance"].to_dict(orient="records"))

        logger.info(
            "Batch explanation completed for %s customers",
            result["total_customers"]
        )

        return {
            "total_customers": result["total_customers"],
            "beeswarm_plot_url": (
                "/explain/batch/plot/beeswarm"
            ),
            "feature_importance_plot_url": (
                "/explain/batch/plot/importance"
            ),
            "feature_importance": feature_importance
        }

    except HTTPException:
        raise

    except (ValueError, TypeError) as error:

        logger.exception(
            "Invalid batch explanation input"
        )

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception(
            "Batch SHAP explanation failed"
        )

        raise HTTPException(
            status_code=500,
            detail="Batch explanation failed"
        )

    finally:
        file.file.close()


# --------------------------------------------------
# BATCH SHAP PLOT RETRIEVAL
# --------------------------------------------------

@app.get("/explain/batch/plot/{plot_type}")
def get_batch_explanation_plot(plot_type: str,_: str = Security(verify_api_key)):

    plot_files = {
        "beeswarm": "batch_shap_beeswarm.png",
        "importance": "batch_shap_feature_importance.png"
    }

    if plot_type not in plot_files:

        raise HTTPException(
            status_code=404,
            detail="Invalid batch explanation plot type"
        )

    plot_path = (
        EXPLANATION_DIR /
        plot_files[plot_type]
    )

    if not plot_path.is_file():

        raise HTTPException(
            status_code=404,
            detail="Batch explanation plot not found"
        )

    return FileResponse(
        path=plot_path,
        media_type="image/png",
        filename=plot_files[plot_type]
    )
