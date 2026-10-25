"""
Step 1 - Data preparation.

Cleans the raw bank export, builds the numeric features (date parts and a
TF-IDF vocabulary of the transaction descriptions), merges the manually
assigned labels and creates the train/test splits.

Note: column names and values (e.g. "Tipo", "Importo", "COMPLETATO") come
from the Italian bank export and are kept as they are.

Run from the repository root:  python src/01_data_preparation.py
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("data/raw.csv")

# Quick inspection of the raw data:
# data.head(); data.shape; data.info(); data.describe()

# --- Cleaning -----------------------------------------------------------
# Keep only completed card payments and bank transfers on the main account.
payment_types = ["Pagamento", "Pagamento con carta"]
new_data = data[data["Tipo"].isin(payment_types)]
new_data = new_data[new_data["State"] == "COMPLETATO"]
new_data = new_data[new_data["Prodotto"] == "Attuale"]
new_data = new_data.drop(columns=["Prodotto", "Costo", "Data di completamento", "State", "Valuta"])

# --- Date features: day, month, hour ------------------------------------
new_data["Data di inizio"] = pd.to_datetime(new_data["Data di inizio"], yearfirst=True)
new_data["Giorno"] = new_data["Data di inizio"].dt.day
new_data["Mese"] = new_data["Data di inizio"].dt.month
new_data["Ora"] = new_data["Data di inizio"].dt.hour
new_data = new_data.drop(columns="Data di inizio")

# --- Transaction type: 1 = bank transfer ("Pagamento"), 0 = card payment ---
new_data["Tipo"] = (new_data["Tipo"] == "Pagamento").astype(int)

# --- Description vocabulary (TF-IDF over the most frequent words) --------
# Tokens are words of at least 2 letters; the vocabulary keeps the 150 (or 100)
# most frequent words that appear in at least 2 descriptions.
vec1 = TfidfVectorizer(token_pattern=r"[a-zà-ù]{2,}", min_df=2, max_features=150)
mat1 = vec1.fit_transform(new_data["Descrizione"])
df1 = pd.DataFrame(mat1.toarray(), columns=vec1.get_feature_names_out(), index=new_data.index)

vec2 = TfidfVectorizer(token_pattern=r"[a-zà-ù]{2,}", min_df=2, max_features=100)
mat2 = vec2.fit_transform(new_data["Descrizione"])
df2 = pd.DataFrame(mat2.toarray(), columns=vec2.get_feature_names_out(), index=new_data.index)

df150 = pd.concat([new_data, df1], axis=1)
df100 = pd.concat([new_data, df2], axis=1)

# Export the datasets sorted by description to make manual labeling easier.
df150.sort_values(by="Descrizione").to_csv("data/150_words_data.csv", index=True)
df100.sort_values(by="Descrizione").to_csv("data/100_words_data.csv", index=True)

# --- Manual labeling ------------------------------------------------------
# data/target.csv has two columns: "index" (row index of the cleaned dataset)
# and "target" (category 0-7). A blank template can be generated with:
#
#   i = df150.sort_values(by="Descrizione").index
#   pd.DataFrame({"index": i, "target": None}).to_csv("data/target.csv", index=False)
#
# and then filled in by hand.
target = pd.read_csv("data/target.csv")
target.index = target["index"]
target = target.drop(columns=["index"])

fulldf_150 = pd.concat([target, df150], axis=1)
fulldf_100 = pd.concat([target, df100], axis=1)

# --- Train/test splits ----------------------------------------------------
# Training sets of 300 and 200 rows (the 200-row set is a subset of the
# 300-row one); everything outside the 300-row set is used as test set.
training300x150 = fulldf_150.sample(300, random_state=42)
training200x150 = training300x150.sample(200, random_state=42)

training300x100 = fulldf_100.sample(300, random_state=42)
training200x100 = training300x100.sample(200, random_state=42)

# Sanity check: the 150-word and 100-word training sets use the same rows.
print(training300x100.index.equals(training300x150.index))
print(training200x100.index.equals(training200x150.index))

test150 = fulldf_150.drop(index=training300x150.index)
test100 = fulldf_100.drop(index=training300x100.index)

fulldf_150.to_csv("processed_data/fulldf_150.csv", index=False)
fulldf_100.to_csv("processed_data/fulldf_100.csv", index=False)
training300x150.to_csv("processed_data/training300x150.csv", index=False)
training200x150.to_csv("processed_data/training200x150.csv", index=False)
training300x100.to_csv("processed_data/training300x100.csv", index=False)
training200x100.to_csv("processed_data/training200x100.csv", index=False)
test150.to_csv("processed_data/test150.csv", index=False)
test100.to_csv("processed_data/test100.csv", index=False)
