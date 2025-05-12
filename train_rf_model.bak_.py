# train_rf_model.py
import os
import mlflow
import dagshub
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, auc, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
