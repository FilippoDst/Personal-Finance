"""
Step 3 - Neural network.

A small fully connected network trained on the text features only
(same inputs as the "vocabulary only" logistic regression model).

NOTE: no random seed is set, so metrics vary slightly from run to run.
Run from the repository root:  python src/03_neural_network.py
"""
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Input
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Non-text features removed from the inputs.
cols = ["Tipo", "Importo", "Saldo", "Giorno", "Mese", "Ora"]

training300x150 = pd.read_csv("processed_data/training300x150.csv")
train_df = training300x150.drop(columns=[*cols, "Descrizione"])
y_train = train_df["target"]
X_train = train_df.drop(columns=["target"])

test150 = pd.read_csv("processed_data/test150.csv")
amount = test150["Importo"]
y_test = test150["target"]
X_test = test150.drop(columns=[*cols, "Descrizione", "target"])

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(50, activation="relu"),
    Dense(25, activation="relu"),
    Dense(15, activation="relu"),
    Dense(8, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy")  # Keras default optimizer
model.fit(X_train, y_train, epochs=700)

probabilities = model.predict(X_test, verbose=0)
p = probabilities.argmax(axis=1)

print("accuracy: ", accuracy_score(y_test, p))
print("precision:", precision_score(y_test, p, average=None))
print("recall:   ", recall_score(y_test, p, average=None))
print("f1:       ", f1_score(y_test, p, average=None))

df_p = pd.DataFrame(p, columns=["prediction"])
df_p.to_csv("Results/NeuralPrediction.csv")

# Total spending per category: predicted vs. true.
cumulative_prediction = pd.concat([df_p, amount], axis=1)
print(cumulative_prediction.groupby("prediction")["Importo"].sum())
print(test150.groupby("target")["Importo"].sum())
