# Import all models here so SQLAlchemy metadata is always fully populated,
# regardless of which model is imported first by application code.
# This prevents NoReferencedTableError when resolving cross-model ForeignKeys.
from app.models.user import User  # noqa: F401
from app.models.prediction import Prediction  # noqa: F401

__all__ = ["User", "Prediction"]
