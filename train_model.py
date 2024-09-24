from joblib import dump
from sklearn.linear_model import LogisticRegression
import numpy as np


if __name__=="__main__":
    data = np.load('data/normed_transformed_data.npy')
    train_indices = np.load('data/train_indices.npy')
    # val_indices = np.load('data/val_indices.npy')
    # test_indices = np.load('data/test_indices.npy')


    cell_file = "data/cells.npy"
    cells = np.load(cell_file, allow_pickle=True).ravel()[0]
    cell_classes = cells["classes"]  # cell classes

    Xtrain = data[train_indices,:]
    Ytrain = cell_classes[train_indices]

    clf = LogisticRegression(random_state=0,
                         penalty='l1',
                         solver='saga',
                         class_weight='balanced',
                         verbose=1
                         ).fit(Xtrain, Ytrain)
    dump(
        clf,
        "data/log_reg_model.joblib",
    )
