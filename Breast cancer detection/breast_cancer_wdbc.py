import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load the data from the uploaded file
# wdbc.data has NO header row. Layout: ID, diagnosis (M/B), then 30 features.
DATA_PATH = "data/wdbc.data"   # change to the full path if the file is elsewhere

base = ["radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
columns = (["id", "diagnosis"]
           + [f"{b}_mean" for b in base]
           + [f"{b}_se" for b in base]
           + [f"{b}_worst" for b in base])

df = pd.read_csv(DATA_PATH, header=None, names=columns)
print("Shape:", df.shape)
print(df["diagnosis"].value_counts())
print("Missing values:", df.isnull().sum().sum())

# Preprocessing: drop the ID (it's not a feature), encode the label
class_names = ["benign", "malignant"]
y = (df["diagnosis"] == "M").astype(int).values   # 1 = malignant, 0 = benign
X = df.drop(columns=["id", "diagnosis"]).values   # 30 numeric features

# 2. Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature scaling (fit on TRAIN only)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build the network
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1, activation="sigmoid"),   # P(malignant)
]) 
model.summary() 

# 5. Compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 6. Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    verbose=1,
)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["loss"], label="train")
ax[0].plot(history.history["val_loss"], label="validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history["accuracy"], label="train")
ax[1].plot(history.history["val_accuracy"], label="validation")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.show()

# 7. Evaluate on unseen test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
print("\nConfusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 8. Predict on new samples (same fitted scaler)
# Stand-in for new data: first 3 raw rows from the file
new_samples = df.drop(columns=["id", "diagnosis"]).values[:3]
probs = model.predict(scaler.transform(new_samples), verbose=0).ravel()
for i, p in enumerate(probs):
    label = int(p > 0.5)
    actual = "malignant" if y[i] == 1 else "benign"
    print(f"Sample {i}: P(malignant) = {p:.3f} -> {class_names[label]} (actual: {actual})")
