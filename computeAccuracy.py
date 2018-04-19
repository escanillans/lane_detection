import pandas as pd


def precision(TP, FP):
    return float(TP/(TP+FP))

def recall(TP,FN):
    return float(TP/(TP+FN))

def f1_score(P, R):
    return float((2*P*R)/(P+R))

# load accuracyResults.csv

df = pd.read_csv('accuracyResults.csv')
TP = df['TP'].sum()
FP = df['FP'].sum()
FN = df['FN'].sum()

precision = precision(TP, FP)
recall = recall(TP, FN)
f1_score = f1_score(precision, recall)

print('precision: '+str(precision))
print('recall: '+ str(recall))
print('F1 Score: '+ str(f1_score))