import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

def confusion_matrix(y_test, predicted_arr, class_dict=None):

    OA_test = round(accuracy_score(y_test, predicted_arr), 4)
    kappa_test = round(cohen_kappa_score(y_test, predicted_arr), 3)

    # confusion matrix
    confusion_pred = pd.crosstab(
        y_test, 
        predicted_arr, 
        rownames=['Actual'], colnames=['Predicted'], 
        margins=True, 
        margins_name="Total"
    )

    for c in np.unique(y_test):
        confusion_pred.loc[c, 'PA']= str(round(
            confusion_pred.loc[c, c] / confusion_pred.loc[c, "Total"], 
            3
        ))
        confusion_pred.loc['UA', c]= str(round(
            confusion_pred.loc[c, c] / confusion_pred.loc["Total", c], 
            3
        ))

    confusion_pred.loc['OA (Holdout)',confusion_pred.columns[0]] = str(round(OA_test, 3))                
    confusion_pred.loc['Kappa',confusion_pred.columns[0]] = str(round(kappa_test, 3)) 

    if class_dict is not None:
        confusion_pred.rename(index=class_dict, columns=class_dict, inplace=True)

    return confusion_pred