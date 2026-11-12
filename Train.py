
import warnings
warnings.filterwarnings("ignore")   # hide warning messages

# Tools for data
import pandas as pd
import pickle, sklearn

 
# The four models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Tools for tuning and pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# Load the two files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = train.drop(columns=["Unnamed: 0", "id"])
test = test.drop(columns=["Unnamed: 0", "id"])

# imputation
col = ["Arrival Delay in Minutes"]

imputer = SimpleImputer(strategy="median")
train[col] = imputer.fit_transform(train[col])
test[col] = imputer.transform(test[col])

# Encoding
cols = ["Gender", "Customer Type", "Type of Travel", "Class", "satisfaction"]

enc = LabelEncoder()
for col in cols:
    train[col] = enc.fit_transform(train[col])
    test[col] = enc.transform(test[col])


# create new feature
rating_cols = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Inflight service", "Cleanliness",
]

train["Average Service Rating"] = train[rating_cols].mean(axis=1)
test["Average Service Rating"] = test[rating_cols].mean(axis=1)


# separate X and Y
X_train = train.drop(columns="satisfaction")
y_train = train["satisfaction"]
X_test = test.drop(columns="satisfaction")
y_test = test["satisfaction"]


# The four models we will use in every stage
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   HistGradientBoostingClassifier(random_state=42),
}


# Pipeline models
pipelines = {}

for name, model in models.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),   # step 1: scale the data
        ("model", model),               # step 2: train the model
    ])

    pipe.fit(X_train, y_train)
    pipelines[name] = pipe


# Hyperparameter  tuning
# The settings we want to try for each model ("model__" means a setting of the model step in the pipeline)
param_grids = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Decision Tree":       {"model__max_depth": [10, 20, None], "model__min_samples_leaf": [1, 5]},
    "Random Forest":       {"model__n_estimators": [100, 200], "model__max_depth": [None, 20]},
    "Gradient Boosting":   {"model__learning_rate": [0.05, 0.1, 0.2], "model__max_iter": [200, 400]},
}

tuned_models = {}

for name, pipe in pipelines.items():
    search = GridSearchCV(pipe, param_grids[name], cv=3, scoring="accuracy")
    search.fit(X_train, y_train)                        # tune using the training data only
    best_model = search.best_estimator_                 # the pipeline with the best settings
    tuned_models[name] = best_model


# save the model

print("scikit-learn version used for saving:", sklearn.__version__)

# 1. Extract the best performing Gradient Boosting pipeline
best_gb_pipeline = tuned_models["Gradient Boosting"]

# 2. Save using standard python serialization protocol
model_filename = "gradient_boosting_pipeline.pkl"
with open(model_filename, "wb") as f:
    pickle.dump(best_gb_pipeline, f)

print(f"Successfully saved the unified pipeline as: {model_filename}")