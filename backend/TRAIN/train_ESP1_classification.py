import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Pour rééquilibrer les classes
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Charger les données CSV
csv_file = "C:/Users/ADAM/Desktop/Nouveau dossier (9)/projet pfe vf/DATA/TRAIN.csv"
df = pd.read_csv(csv_file, delimiter=';')

# Nettoyage des colonnes de latitude et longitude
df['LONG_DD'] = df['LONG_DD'].str.replace(',', '.').astype(float)
df['LAT_DD'] = df['LAT_DD'].str.replace(',', '.').astype(float)

# Nettoyage des colonnes environnementales (remplacement des virgules par des points)
df['Salinite'] = df['Salinite'].str.replace(',', '.').astype(float)
df['Temp'] = df['Temp'].str.replace(',', '.').astype(float)
df['DO'] = df['DO'].str.replace(',', '.').astype(float)
df['pH'] = df['pH'].str.replace(',', '.').astype(float)

# Sélection des colonnes pertinentes
features = ['LAT_DD', 'LONG_DD', 'Salinite', 'Temp', 'DO', 'pH']
target = 'Pres_Esp1'

X = df[features]
y = df[target]

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardisation des caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Rééquilibrage des classes avec SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Optimisation des hyperparamètres avec GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Meilleurs paramètres et meilleur modèle
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Entraînement du modèle avec les meilleurs paramètres
best_model.fit(X_train_resampled, y_train_resampled)

# Prédictions sur le jeu de test
y_pred = best_model.predict(X_test_scaled)

# Évaluation du modèle
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Sauvegarde du modèle
joblib.dump(best_model, 'species_prediction_ESP1_optimized.pkl')

# Analyse des importances des caractéristiques
importances = best_model.feature_importances_
plt.bar(features, importances, color='skyblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()
