"""
Step 2 - Logistic regression models.

Trains the four baseline models (300/200 training rows x 150/100 vocabulary
words) and the "vocabulary only" model (no amount, balance or date features),
then evaluates them on the test set.

Run from the repository root:  python src/02_logistic_regression.py
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

RUN_ALL_MODELS = True  # False = train only the "vocabulary only" model

training200x150 = pd.read_csv("processed_data/training200x150.csv")
training300x150 = pd.read_csv("processed_data/training300x150.csv")
training200x100 = pd.read_csv("processed_data/training200x100.csv")
training300x100 = pd.read_csv("processed_data/training300x100.csv")
test150 = pd.read_csv("processed_data/test150.csv")
test100 = pd.read_csv("processed_data/test100.csv")
fulldf_150 = pd.read_csv("processed_data/fulldf_150.csv")

drop_cols = ["target", "Descrizione"]
y_test150, X_test150 = test150["target"], test150.drop(columns=drop_cols)
y_test100, X_test100 = test100["target"], test100.drop(columns=drop_cols)

# Non-text features, removed in the "vocabulary only" model.
cols = ["Tipo", "Importo", "Saldo", "Giorno", "Mese", "Ora"]

# --- Vocabulary-only model (300 rows, 150 words, text features only) -----
model_voc = LogisticRegression(max_iter=15000)
voc_train = training300x150.drop(columns=[*cols, "Descrizione"])
yvoc_train = voc_train["target"]
Xvoc_train = voc_train.drop(columns=["target"])
model_voc.fit(Xvoc_train, yvoc_train)
Xvoc_test = X_test150.drop(columns=cols)

models = {"vocabulary only (300/150)": (model_voc, Xvoc_train, yvoc_train, Xvoc_test, y_test150)}

# --- The four baseline models ---------------------------------------------
if RUN_ALL_MODELS:
    for name, train, X_te, y_te in [
        ("model 1: 300 examples / 150 words", training300x150, X_test150, y_test150),
        ("model 2: 200 examples / 150 words", training200x150, X_test150, y_test150),
        ("model 3: 300 examples / 100 words", training300x100, X_test100, y_test100),
        ("model 4: 200 examples / 100 words", training200x100, X_test100, y_test100),
    ]:
        X_tr, y_tr = train.drop(columns=drop_cols), train["target"]
        m = LogisticRegression(max_iter=15000).fit(X_tr, y_tr)
        models[name] = (m, X_tr, y_tr, X_te, y_te)


def evaluate(m, Xtr, ytr, Xte, yte):
    p = m.predict(Xte)
    return {
        "acc_train": m.score(Xtr, ytr),
        "acc_test": m.score(Xte, yte),
        "precision_by_class": precision_score(yte, p, average=None),
        "recall_by_class": recall_score(yte, p, average=None),
        "f1_by_class": f1_score(yte, p, average=None),
    }


for name, args in models.items():
    print(name)
    print(evaluate(*args))

# Predictions of the vocabulary-only model, used by the visualization step.
pred = model_voc.predict(Xvoc_test)
pd.DataFrame(pred, columns=["pred"]).to_csv("Results/LogisticPrediction.csv")

# Error analysis for class 7 (Savings): rows predicted as 7, then all true
# class-7 rows in the full dataset and in the training set.
print(test150[pred == 7])
print(fulldf_150[fulldf_150["target"] == 7])
print(training300x150[training300x150["target"] == 7])
