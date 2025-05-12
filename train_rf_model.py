# train_rf_model.py
import os

import dagshub
import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import (
    LabelBinarizer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
