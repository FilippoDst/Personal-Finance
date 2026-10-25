"""
Step 4 - Visualization.

Compares total spending per category (true labels vs. neural network vs.
logistic regression) and saves the two figures used in the README.

Run from the repository root:  python src/04_visualization.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("processed_data/test150.csv")  # columns "target" and "Importo"
LogisticPrediction = pd.concat([pd.read_csv("Results/LogisticPrediction.csv"), df["Importo"]], axis=1)
NeuralPrediction = pd.concat([pd.read_csv("Results/NeuralPrediction.csv"), df["Importo"]], axis=1)

categories = ["Groceries", "Clothing", "Travel", "Food & Drinks",
              "Health", "Work supplies", "Misc.", "Savings"]

# Total spent per category; categories never predicted by a model count as 0.
true_spend = df.groupby("target")["Importo"].sum().abs().reindex(range(8), fill_value=0)
nn_spend = NeuralPrediction.groupby("prediction")["Importo"].sum().abs().reindex(range(8), fill_value=0)
lr_spend = LogisticPrediction.groupby("pred")["Importo"].sum().abs().reindex(range(8), fill_value=0)

# --- Figure 1: total amount per category ----------------------------------
x = np.arange(len(categories))
width = 0.25

plt.figure(figsize=(16, 8))
plt.bar(x - width, true_spend, width, label="True labels", color="#1F3A5F")
plt.bar(x, nn_spend, width, label="Neural network", color="#2F6FA7")
plt.bar(x + width, lr_spend, width, label="Logistic regression", color="#1E7B34")
plt.xticks(x, categories, rotation=30)
plt.xlabel("Spending category")
plt.ylabel("Amount (€)")
plt.legend()
plt.tight_layout()
plt.savefig("docs/images/amount_by_category.png", dpi=150, bbox_inches="tight")
plt.show()

# --- Figure 2: percentage share of each category --------------------------
series = [true_spend, nn_spend, lr_spend]
names = ["True labels", "Neural network", "Logistic regression"]

fig, ax = plt.subplots(figsize=(14, 3))
for row, values in enumerate(series):
    share = values / values.sum() * 100
    left = 0
    for i in range(8):
        ax.barh(row, share[i], left=left, color=plt.cm.tab10(i),
                label=categories[i] if row == 0 else None)
        left += share[i]

ax.set_yticks(range(3), names)
ax.set_xlabel("Share of total spending (%)")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))  # below the plot
fig.savefig("docs/images/spending_share.png", dpi=150, bbox_inches="tight")
plt.show()
