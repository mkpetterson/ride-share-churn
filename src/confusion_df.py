import pandas as pd
from sklearn.metrics import confusion_matrix
import six
import matplotlib.pyplot as plt
import numpy as np


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

def render_mpl_table(data, col_width=4.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
            
    return fig, ax