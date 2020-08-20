# TanD.util module

# Index:
  * [Trainer](#class-Trainer)
  * [get_request_features](#func-get_request_features)
  * [get_preds_labels_scores](#func-get_preds_labels_scores)
  * [eval_model_binary](#func-eval_model_binary)
  * [eval_model_per_class](#func-eval_model_per_class)
  * [save_confusion_matrix](#func-save_confusion_matrix)
  * [save_roc_curve](#func-save_roc_curve)
  * [save_pr_curve](#func-save_pr_curve)
  * [is_categorical](#func_is_categorical)
  
  * [get_percentage_label_not](#get_percentage_label_not)
  * [save_per_feature_analysis](#save_per_feature_analysis)
  * [eval_classification_model_predictions_per_feature](#eval_classification_model_predictions_per_feature)
  
  
  
---
## class Trainer
### tand.util.Trainer(model, device, criterion)

Implements trainer for PyTorch models, in order to ease `tand` no-code workflows
Parameters:
 * model `torch.nn.module` -> PyTorch model to be trained by trainer;
 * device `torch.device` -> Device where the training will occur (`cpu` or `cuda`);
 * criterion `torch.nn.module` or `function` -> Object of which the call method receive two `torch.Tensor` parameters and output corresponds to the desired loss function.

Method `train(dataloader_train, dataloader_test, epochs, log_every)` -> Trains the model and validates it on `dataloader_test` for `epochs`, logging `log_every` steps in the screen.

---
## func get_request_features
### tand.util.get_request_features(df)
Parameters:
 * `df` `pandas.DataFrame` -> Dataframe corresponding to feature table without labels column
 
Returns `dict` with keys, in order, corresponding to the column features of `df` to be used as a request body model in the workflow.


---
## func get_preds_labels_scores
### tand.util.get_preds_labels_scores(dataloader, classifier, device)

Parameters:
 * dataloader `torch.utils.data.DataLoader` dataloader on which the model will be evaluated
 * classifier `torch.nn.module` model to be evaluated
 * device `torch.device` device where the operations shall occur
 
 Returns `torch.Tensor`, `torch.Tensor`, `torch.Tensor` corresponding to labels, predicted classes and predicted classes scores, each with shape `[n_samples, ]`


---
## func eval_model_binary
### tand.util.eval_model_binary(y_true, y_pred, label_name="")
Parameters:
 * y_true `numpy.array` corresponding to true labels of a class (1 if corresponds to label name, 0 in doesn`t) with shape `[n_samples, ]`
 * y_pred `numpy.array` corresponding to true labels of a class (1 if corresponds to label name, 0 in doesn`t) with shape `[n_samples, ]`
 * label_name `str` corresponding to label name
 
 Returns `dict` with metrics of precision, recall, specificity, fpr and np for class label_name

---
## func eval_model_per_class
### tand.util.eval_model_per_class(y_true, y_pred, label_names)
Parameters:
 * y_true `numpy.array` corresponding to true labels of validation dataset with shape `[n_samples, ]`
 * y_pred `numpy.array` corresponding to predicted labels of validation dataset with shape `[n_samples, ]`
 * label_names `list` with items corresponding of label name represented by its index
 

Returns `dict` with metrics of precision, recall, specificity, fpr and np for each class by performing `eval_model_binray` with data corresponding of binary classification scores to each class

---
## func save_confusion_matrix
### tand.util.save_confusion_matrix(y_true, y_pred, label_names, fname="confusion_matrix.png")
Parameters:
 * y_true `numpy.array` corresponding to true labels of validation dataset with shape `[n_samples, ]`
 * y_pred `numpy.array` corresponding to predicted labels of validation dataset with shape `[n_samples, ]`
 * label_names `list` with items corresponding of label name represented by its index
 * fname `str` path where the image will be saved, works with `.png` format
 
Saves confusion matrix beautiful plot with seaborn to path string fname, then returns string fname (helps automatic training and evaluation)

---
## func save_roc_curve
### tand.util.save_roc_curve(y_true, scores, label_names, fname="roc_curve.png")

Parameters:
 * y_true `numpy.array` corresponding to true labels of validation dataset with shape `[n_samples, ]`
 * scores `numpy.array` corresponding to predicted scores of validation dataset with shape `[n_samples, ]`
 * label_names `list` with items corresponding of label name represented by its index
 * fname `str` path where the image will be saved, works with `.png` format
 
Saves roc curve plot for each class on the same image then returns string fname
 

---
## func save_pr_curve
### tand.util.save_pr_curve(y_true, scores, label_names, fname="pr_curve.png")

Parameters:
 * y_true `numpy.array` corresponding to true labels of validation dataset with shape `[n_samples, ]`
 * scores `numpy.array` corresponding to predicted scores of validation dataset with shape `[n_samples, ]`
 * label_names `list` with items corresponding of label name represented by its index
 * fname `str` path where the image will be saved, works with `.png` format
 
Saves Precision-Recall curve plot for each class on the same image then returns string fname

---
## func is_categorical
### tand.util.is_categorical(df, feature)

Parameters:
 * df `pandas.DataFrame`
 * feature `str`
 
 Returns bool `True` if feature is categorical and binary on df else `False`
 
 ---
 
 ## func get_percentage_label_not
 ### tand.util.get_percentage_label_not(df, feature, label, label_column))
 
 Parameters:
  * df `pandas.DataFrame`
  * feature `str`
  * label `int`
  * label_column `str`
 
 Assumes that feature is categorical and binary
 
 Returns `dict` containing, the proportion of df column feature=1 which has label=1 and label=0 and the proportion of df column feature=1 which has label=1 and label=0 on dict with shape:
 ```python
{
        f"{label}": {
            'positive': positive_percentage_label,
            'negative': 1 - positive_percentage_label
        },
        f"not_{label}": {
            'positive': positive_percentage_not_label,
            'negative': 1 - positive_percentage_not_label
        },
    }
```

---

## func get_all_categoric_percentages
### tand.util.get_all_categoric_percentages(df, features, label, label_column)

Parameters:
 * df `pandas.DataFrame` which contains the data
 * features `list` list of all categoric features
 * label `int` index of label which the feature proportions will be evaluated
 * label_column `str` index of the column which contains the labels
 
Performs `tand.util.get_percentage_label_not` on all categorical features and returns `dict` mapping the feature name to the `dict` that results from `tand.util.get_percentage_label_not`

---

## func save_per_feature_analysis
### tand.util.save_per_feature_analysis(keys, predicted_label_name, predicted_label_names, label_idx, prediction_info, label_column, df)

Parameters:
 * keys `list` features of df
 * predicted_label_name
 * predicted_label_names
 * label_idx
 * prediction_info `dict` that results from `tand.util.get_all_categoric_percentages`
 * label_column `str`
 * df `pandas.DataFrame`
 
 
 Saves pie charts using the data from each of predicition_info sub_dicts (proportion of labels given a feature) and `seaborn.distplot`s comparing all the continuous features over all classes on path 
 ```python 
 f"{predicted_label_name}_per_feature_analysis.png"
```
and returns this path. Helps debugging model and checking for bad-data.

---
## func eval_classification_model_predictions_per_feature
### tand.util.eval_classification_model_predictions_per_feature(df_filename, classifier, label_column_name, label_names, to_drop, preprocess=lambda i: i, use_torch=False, device=None)

Paremeters:
 * df_filename `str` filename corresponding to `.csv` file
 * classifier `torch.nn.Module` or any model from `sklearn`
 * label_column_name `str`corresponding to a column where the predictions of the model will be put in the `pandas.DataFrame` generated from df_filename
 * label_names `list` of `str` containing the label names
 * to_drop `list` corresponding to the columns to be dropped of the model (not the labels one)
 * preprocess `function` used to preprocess data at train step
 * use_torch `bool` checking if classifier uses `torch`
 * device `None` or `torch.device`
 
 Creates and preprocess dataframe on path df_filename, then performs predictions in all models and performs `tand.util.save_per_feature_analysis` for all features, saving the plots and returning its paths. Intended to be used on `tand` template to store files on `mlflow` and then delete them locally. 
 ---
 
 ###### Made by Pi Esposito