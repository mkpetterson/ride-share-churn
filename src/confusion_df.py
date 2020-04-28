import pandas as pd
from sklearn.metrics import confusion_matrix


def confusion_df(y_true, y_pred):
    """ Creates confusion matrix in pandas dataframe
    C[0,0] = True Negatives
    C[0,1] = False Positives
    C[1,0] = False Negatives
    C[1,1] = True Positives
    
    Inputs
    actual labels: np array
    predictd labels: np array
    
    Returns
    pandas dataframe
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Put into pandas dataframe
    confusion = pd.DataFrame({'Predicted Negative': [tn, fn], 'Predicted Positive': [fp, tp]}, 
                             index=['Actual Negative', 'Actual Positive']) 
      
    return confusion