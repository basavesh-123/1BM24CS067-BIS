
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("intrusion_dataset.csv")

X = pd.get_dummies(data.drop("label", axis=1))
y = data["label"]

# Convert labels into numbers
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Grey Wolf Optimization
n_wolves = 5
iterations = 10
n_features = X_train.shape[1]

wolves = np.random.randint(
    0, 2, (n_wolves, n_features)
)

def fitness(wolf):

    selected = np.where(wolf == 1)[0]

    if len(selected) == 0:
        return 0

    model = RandomForestClassifier(
        n_estimators=20, random_state=42
    )

    model.fit(X_train.iloc[:, selected], y_train)

    prediction = model.predict(
        X_test.iloc[:, selected]
    )

    return accuracy_score(y_test, prediction)

# Optimization process
for i in range(iterations):

    scores = [fitness(w) for w in wolves]

    alpha = wolves[np.argmax(scores)].copy()

    for j in range(n_wolves):

        r = np.random.rand(n_features)

        wolves[j] = np.where(
            r < 0.5,
            alpha,
            wolves[j]
        )

    print("Iteration:", i + 1,
          "Best Accuracy:", max(scores))

# Select best features
best_wolf = alpha
selected_features = np.where(best_wolf == 1)[0]

print("\nSelected Features:")
print(X.columns[selected_features].tolist())

# Final classification
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train.iloc[:, selected_features], y_train)

predictions = model.predict(
    X_test.iloc[:, selected_features]
)

accuracy = accuracy_score(y_test, predictions)

print("\nIntrusion Detection Accuracy:",
      accuracy * 100, "%")

print("Normal and Attack Classification Completed")
