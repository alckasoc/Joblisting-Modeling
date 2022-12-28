# Imports.
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomRemover(BaseEstimator, TransformerMixin):
    
    def __init__(self, useless_attribs):
        self.useless_attribs = useless_attribs
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.drop(self.useless_attribs, axis=1)
        return X_copy
    
class CatGrouper(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols, threshold=0.1):
        self.cols = cols
        self.threshold = threshold
        self.cat_groups = {}
        
    def fit(self, X, y=None):
        for attrib in self.cols:
            vc = X[attrib].value_counts(normalize=True)
            thres = vc < self.threshold
            keep = vc[np.logical_not(thres)].index
            self.cat_groups[attrib] = list(keep)
        return self
    
    def _map_func(self, v, attrib):
        if v not in self.cat_groups[attrib]:
            v = "Other"
        return v
    
    def transform(self, X):
        X_copy = X.copy()
        for attrib in self.cols:
            X_copy[attrib] = X_copy[attrib].apply(self._map_func, attrib=attrib)
        return X_copy

def get_pred_ranked_avg(models, ranking, X_test_prep, y_test=None):
    assert len(models) == len(ranking)
    
    all_y_test_pred = []

    for m, model_folds in models.items():
        model_preds = []
        if m == "xgboost_evalml":
            # Hard-coded in.
            ind = [0, 2, 3, 5, 8, 9, 10, 11, 15, 17, 22, 24, 25, 28, 29]
            X_test_prep_select = X_test_prep[:, ind]
        for model_fold in model_folds:
            if m == "xgboost_evalml":
                y_test_pred_evalml = model_fold.predict(X_test_prep_select)
            else:
                y_test_pred = model_fold.predict(X_test_prep)
                model_preds.append(y_test_pred)
            
        if m != "xgboost_evalml":
            model_preds = np.mean(np.array(model_preds), axis=0)            
            all_y_test_pred.append(model_preds)
        
    all_y_test_pred = np.array(all_y_test_pred)
    
    # Concatenate our ensemble with EvalML's XGBoost.
    all_y_test_pred = np.concatenate((all_y_test_pred, np.expand_dims(y_test_pred_evalml, 0)), axis=0)

    return np.sum(all_y_test_pred * ranking[:, np.newaxis], 0)