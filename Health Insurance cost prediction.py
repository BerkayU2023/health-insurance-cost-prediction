import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("insurance.csv")

print(df.info())
print(df.head(23))

num_cols = df.select_dtypes(exclude = "object")
obj_cols = df.select_dtypes(include = "object")

print(obj_cols)
print(num_cols)

df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)

print(df.head(23))

df["smoker"] = df["smoker"].map({"no" : 0, "yes" : 1})
df["sex"] = df["sex"].map({"female" : 0, "male" : 1})

print(df.head(23))
print(df.info())
print(df.describe())


X = df.drop("charges", axis = 1)
y = df["charges"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

X_train_scaled = minmax.fit_transform(X_train)
X_test_scaled = minmax.transform(X_test)

X_cols = minmax.get_feature_names_out()

X_train = pd.DataFrame(X_train_scaled, columns = X_cols)
X_test = pd.DataFrame(X_test_scaled, columns = X_cols)

print(X_train)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2 = r2_score(true, pred)

    return mae, mse, rmse, r2

models = {
    "Linear" : LinearRegression(),
    "SVR" : SVR(),
    "DecisionTree" : DecisionTreeRegressor(),
    "RandomForest" : RandomForestRegressor(),
    "Adaboost" : AdaBoostRegressor(),
    "Gradient" : GradientBoostingRegressor(),
    "XGBoost" : XGBRegressor(),
    "LightGBM" : LGBMRegressor()
}

results = []
t_results = []
merge = []

for name, model in models.items():
    model = model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    test_mae, test_mse, test_rmse, test_r2 = evaluate_model(y_test, y_pred)
    train_mae, train_mse, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)

    print(f" \n {name} Results")
    print(f"Test MAE  : {test_mae:.4f}      | Train MAE  : {train_mae:.4f}")
    print(f"Test MSE  : {test_mse:.4f}  | Train MSE  : {train_mse:.4f}")
    print(f"Test RMSE : {test_rmse:.4f}      | Train RMSE : {train_rmse:.4f}")
    print(f"Test R2   : {test_r2:.4f}         | Train R2   : {train_r2:.4f}")

    if train_r2 - test_r2 > 0.15:
        print("Overfitting olabilir!","\n")

    print("_" * 50)


    merge.append([name, test_r2, train_r2])


merge_df = pd.DataFrame(merge, columns = ["Model", "Test R2 Score","Train R2 Score"]).sort_values(by = "Test R2 Score", ascending = False)

print(merge_df)


# HYPERPARAMETER TUNİNG FOR GRADIENT, LGBM, RANDOMFOREST, DECISION TREE
from sklearn.model_selection import GridSearchCV
print("\n", "HYPERPARAMETER TUNING", "\n")
param_dtree = {
    "max_depth" : [3, 5, 7, 10, 15],
    "min_samples_split" : [2, 10, 20],
    "min_samples_leaf" : [1, 5, 10, 20]
}

grid_dtree = GridSearchCV(estimator = DecisionTreeRegressor(random_state=23), param_grid = param_dtree, cv=5, scoring="r2", n_jobs = -1)

grid_dtree.fit(X_train ,y_train)

print("Best Parameters: ", grid_dtree.best_params_)

best_tree = grid_dtree.best_estimator_

y_test_pred_tuned = best_tree.predict(X_test)
y_train_pred_tuned = best_tree.predict(X_train)

test_mae, test_mse, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred_tuned)
train_mae, train_mse, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred_tuned)

print(f"\n Optimized Decision Tree Results")
print(f"Test R2   : {test_r2:.4f}         | Train R2   : {train_r2:.4f}")
print(f"Test RMSE : {test_rmse:.4f}      | Train RMSE : {train_rmse:.4f}")

if train_r2 - test_r2 > 0.15:
    print("Overfitting devam ediyor!", "\n")
else:
    print("Overfitting engellendi.", "\n")

# İLERİ SEVİYE MODELLER İÇİN HİPERPARAMETRE OPTİMİZASYONU
from sklearn.model_selection import GridSearchCV

print("\n", "--- ILERI SEVIYE HIPERPARAMETRE OPTIMIZASYONU ---", "\n")


tuning_models = {
    "RandomForest": RandomForestRegressor(random_state=23),
    "XGBoost": XGBRegressor(random_state=23),
    "LightGBM": LGBMRegressor(random_state=23, verbose = -1)
}

tuning_grids = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "min_samples_split": [5, 10]
    },
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1]
    },
    "LightGBM": {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "num_leaves": [20, 31, 40]
    }
}

tuned_results = []


for name in tuning_models.keys():
    print(f"[{name}] modeli optimize ediliyor...")
   
    grid_search = GridSearchCV(estimator = tuning_models[name], param_grid = tuning_grids[name], cv=5, scoring="r2", n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    print(f"En Iyi Parametreler ({name}): {grid_search.best_params_}")
    
    # Bulunan en iyi modelle tahmin yapıyoruz
    best_model = grid_search.best_estimator_
    
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    test_mae, test_mse, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)
    train_mae, train_mse, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)
    
    print(f"Test R2   : {test_r2:.4f}  | Train R2   : {train_r2:.4f}")
    print(f"Test RMSE : {test_rmse:.4f}  | Train RMSE : {train_rmse:.4f}")
    
    if train_r2 - test_r2 > 0.15:
        print("Durum: Overfitting (Asiri Ogrenme) tam cozulemedi, parametreler daraltilabilir.")
    else:
        print("Durum: Basarili")
        
    print("-" * 60)
    
    tuned_results.append([name, test_r2, train_r2])


tuned_df = pd.DataFrame(tuned_results, columns=["Model", "Tuned Test R2", "Tuned Train R2"]).sort_values(by="Tuned Test R2", ascending=False)

print("\n--- OPTIMIZE EDILMIS MODELLERIN SIRALAMASI ---")
print(tuned_df)

print("\n", "--- GRADIENT BOOSTING HIPERPARAMETRE OPTIMIZASYONU ---", "\n")

param_gb = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
    "min_samples_split": [2, 5]
}

grid_gb = GridSearchCV(estimator = GradientBoostingRegressor(random_state=23), param_grid = param_gb, cv=5, scoring = "r2", n_jobs = -1)

grid_gb.fit(X_train, y_train)

print("Gradient Boost Best Parameters", grid_gb.best_params_)

best_gb = grid_gb.best_estimator_

y_test_pred_gb = best_gb.predict(X_test)
y_train_pred_gb = best_gb.predict(X_train)

test_mae, test_mse, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred_gb)
train_mae, train_mse, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred_gb)

print(f" Test RMSE : {test_rmse:.4f} | Train RMSE : {train_rmse:.4f}")
print(f" Test R2 :   {test_r2:.4f}    | Train R2 : {train_r2:.4f}")

if train_r2 - test_r2 > 0.15:
    print("Durum: Overfitting (Asiri Ogrenme) tam cozulemedi, parametreler daraltilabilir.")
else:
    print("Durum: Basarili")