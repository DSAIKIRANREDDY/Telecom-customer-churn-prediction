
import joblib
import shap

from pathlib import Path
from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class ModelLoader:

    def __init__(self, model_path: Path):

        self.model_path = Path(model_path)

        self.model = None
        self.preprocessor = None
        self.classifier = None
        self.explainer = None


    def load_model(self):

        try:

            logger.info("Loading production model")

            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}"
                )

            self.model = joblib.load(self.model_path)

            # Extract fitted pipeline components
            self.preprocessor = self.model.named_steps["preprocessor"]

            self.classifier = self.model.named_steps["classifier"]

            logger.info("Model loaded successfully")

            return self.model

        except Exception:

            logger.exception("Model loading failed")
            raise


    def load_explainer(self):

        try:

            if self.classifier is None:
                raise RuntimeError(
                    "Load the model before initializing SHAP"
                )

            logger.info("Initializing SHAP TreeExplainer")

            self.explainer = shap.TreeExplainer(
                self.classifier
            )

            logger.info("SHAP explainer initialized successfully")

            return self.explainer

        except Exception:

            logger.exception("SHAP explainer initialization failed")
            raise

