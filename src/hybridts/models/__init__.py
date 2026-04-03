"""Base model implementations and factories."""
from .classic import arima_forecast, ets_forecast, auto_arima_forecast, prophet_forecast
from .factory import make_model
from .autoformer import Autoformer
from .dlinear import DLinear
from .fedformer import FEDformer
from .helformer_pt import HelformerPT, helformer_forecast_pt
from .patchtst import PatchTST
from .nbeats import NBEATSV2
from .timesnet import TimesNetV2

# Helformer requires TensorFlow — import lazily to avoid crashing when TF is absent
# or broken.  Users who need helformer should import directly:
#   from hybridts.models.helformer import helformer_forecast

__all__ = [
    "make_model",
    "Autoformer",
    "DLinear",
    "FEDformer",
    "HelformerPT",
    "helformer_forecast_pt",
    "PatchTST",
    "NBEATSV2",
    "TimesNetV2",
    "arima_forecast",
    "ets_forecast",
    "auto_arima_forecast",
    "prophet_forecast",
]
