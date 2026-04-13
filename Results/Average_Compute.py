import numpy as np

def stats(arr):
    return round(np.mean(arr), 4), round(np.std(arr, ddof=1), 4)

#EHO Results
eho_acc = [0.94, 0.93, 0.93, 0.94, 0.92, 0.94, 0.93, 0.90, 0.93, 0.94]

eho_prec_h = [0.80, 0.84, 0.85, 0.86, 0.79, 0.82, 0.83, 0.72, 0.78, 0.85]

eho_prec_d = [0.96, 0.94, 0.94, 0.95, 0.93, 0.96, 0.94, 0.93, 0.95, 0.95]

eho_macro_f1 = [0.87, 0.85, 0.85, 0.86, 0.81, 0.87, 0.84, 0.79, 0.85, 0.87]

eho_weighted_f1 = [0.93, 0.93, 0.93, 0.93, 0.91, 0.94, 0.92, 0.90, 0.93, 0.94]

eho_roc = [0.9641, 0.9588, 0.9611, 0.9598, 0.9502, 0.9663, 0.9487, 0.9288, 0.9564   , 0.9697]

#DBO Results
dbo_acc = [0.85, 0.88, 0.85, 0.85, 0.87, 0.86, 0.85, 0.86, 0.89, 0.86]

dbo_prec_h = [0.65, 0.79, 0.50, 0.58, 0.74, 0.70, 0.57, 0.63, 0.72, 0.68]

dbo_prec_d = [0.86, 0.88, 0.85, 0.86, 0.88, 0.86, 0.86, 0.86, 0.90, 0.87]

dbo_macro_f1 = [0.49, 0.66, 0.46, 0.49, 0.63, 0.52, 0.53, 0.52, 0.72, 0.58]

dbo_weighted_f1 = [0.79, 0.85, 0.78, 0.79, 0.84, 0.80, 0.81, 0.80, 0.87, 0.82]

dbo_roc = [0.8106, 0.8813, 0.7859, 0.8350, 0.8212, 0.8015, 0.8250, 0.8264, 0.9014, 0.8510]

#SMA Results
sma_acc = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.86, 0.85]

sma_prec_h = [0.00, 0.00, 0.50, 0.00, 0.00, 0.73, 0.00, 0.00, 0.66, 0.00]

sma_prec_d = [0.85, 0.85, 0.85, 0.85, 0.85, 0.86, 0.85, 0.85, 0.86, 0.85]

sma_macro_f1 = [0.46, 0.46, 0.46, 0.46, 0.46, 0.49, 0.46, 0.46, 0.52, 0.46]

sma_weighted_f1 = [0.78, 0.78, 0.78, 0.78, 0.78, 0.79, 0.78, 0.78, 0.80, 0.78]

sma_roc = [0.5125, 0.6583, 0.6597, 0.6734, 0.7002, 0.5035, 0.6980, 0.7784, 0.5873, 0.6996]

print("EHO:")
print("Accuracy:", stats(eho_acc))
print("Precision Healthy:", stats(eho_prec_h))
print("Precision Diseased:", stats(eho_prec_d))
print("Macro F1:", stats(eho_macro_f1))
print("Weighted F1:", stats(eho_weighted_f1))
print("ROC-AUC:", stats(eho_roc))

print("\nDBO:")
print("Accuracy:", stats(dbo_acc))
print("Precision Healthy:", stats(dbo_prec_h))
print("Precision Diseased:", stats(dbo_prec_d))
print("Macro F1:", stats(dbo_macro_f1))
print("Weighted F1:", stats(dbo_weighted_f1))
print("ROC-AUC:", stats(dbo_roc))

print("\nSMA:")
print("Accuracy:", stats(sma_acc))
print("Precision Healthy:", stats(sma_prec_h))
print("Precision Diseased:", stats(sma_prec_d))
print("Macro F1:", stats(sma_macro_f1))
print("Weighted F1:", stats(sma_weighted_f1))
print("ROC-AUC:", stats(sma_roc))