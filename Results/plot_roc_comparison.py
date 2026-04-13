import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load saved predictions
y_true = np.load("roc_ytrue.npy")
y_eho = np.load("roc_eho.npy")
y_dbo = np.load("roc_dbo.npy")
y_sma = np.load("roc_sma.npy")

# Compute ROC
fpr_eho, tpr_eho, _ = roc_curve(y_true, y_eho)
fpr_dbo, tpr_dbo, _ = roc_curve(y_true, y_dbo)
fpr_sma, tpr_sma, _ = roc_curve(y_true, y_sma)

auc_eho = auc(fpr_eho, tpr_eho)
auc_dbo = auc(fpr_dbo, tpr_dbo)
auc_sma = auc(fpr_sma, tpr_sma)

# Plot
plt.figure(figsize=(8,6), dpi=300)
plt.plot(fpr_eho, tpr_eho, label=f"EHO (AUC = {auc_eho:.4f})")
plt.plot(fpr_dbo, tpr_dbo, label=f"DBO (AUC = {auc_dbo:.4f})")
plt.plot(fpr_sma, tpr_sma, label=f"SMA (AUC = {auc_sma:.4f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Comparison of EHO vs DBO vs SMA")
plt.legend()
plt.grid(True)

plt.savefig("ROC_Comparison_300DPI.png", dpi=300, bbox_inches='tight')
plt.show()
