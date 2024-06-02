# Importing necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Defining features and target variable
X = df_tr
y = df['diagnosis']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.21, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building a Sequential neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the neural network model
model.fit(X_train_scaled, y_train, epochs=60, batch_size=16, validation_split=0.21)

# Evaluating the model on test data
loss, accuracy = model.evaluate(X_test_scaled, y_test)

# Predicting using the neural network model
model.predict([[0, 0, 2, 0, 2.984651, 48.119630, 269.088550]])

# Training a Support Vector Machine (SVM) model with RBF kernel
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Evaluating the SVM model
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy (SVM - RBF):", accuracy)

# Training a Support Vector Machine (SVM) model with sigmoid kernel
svm_model = SVC(kernel='sigmoid', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Evaluating the SVM model
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy (SVM - Sigmoid):", accuracy)

# Training an XGBoost model
y_train_encoded = y_train - 1
y_test_encoded = y_test - 1
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train_encoded)

# Evaluating the XGBoost model
y_pred_encoded = xgb_model.predict(X_test_scaled)
y_pred = y_pred_encoded + 1
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy (XGBoost):", accuracy)

# Training a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluating the Random Forest model
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy (Random Forest):", accuracy)

# Training a Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)

# Evaluating the Gradient Boosting model
y_pred = gb_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy (Gradient Boosting):", accuracy)

# Tuning hyperparameters for the Gradient Boosting model using GridSearchCV
gb_model2 = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_search = GridSearchCV(estimator=gb_model2, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Selecting the best model from GridSearchCV results
best_model = grid_search.best_estimator_

# Evaluating the best model
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy (Gradient Boosting - Tuned):", accuracy)

# Saving the trained Gradient Boosting model to a file
joblib.dump(gb_model, 'gradient_boosting_model.pkl')
