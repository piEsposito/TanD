# TanD.util module
They all inherit from torch.nn.Module
# Index:
  * [Trainer](#class-Trainer)
  * [get_request_features](#func-get_request_features)
  * [get_preds_labels_scores](#func-get_preds_labels_scores)
  * [eval_model_binary](#func-eval_model_binary)
  * [eval_model_per_class](#func-eval_model_per_class)
  * [save_confusion_matrix](#func-save_confusion_matrix)
  * [save_roc_curve](#func-save_roc_curve)
  * [save_pr_curve](#func-save_pr_curve)
  
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
 
Saves confusion matrix beautiful plot with seaborn to path fname, then returns fname (helps automatic training and evaluation)

---
## func save_roc_curve
### tand.util.save_roc_curve(y_true, scores, label_names, fname="roc_curve.png")

Parameters:
 * y_true `numpy.array` corresponding to true labels of validation dataset with shape `[n_samples, ]`
 * scores `numpy.array` corresponding to predicted scores of validation dataset with shape `[n_samples, ]`
 * label_names `list` with items corresponding of label name represented by its index
 * fname `str` path where the image will be saved, works with `.png` format
 
Saves roc curve plot for each class on the same image then returns fname
 

---
## func save_pr_curve
### tand.util.save_pr_curve(y_true, scores, label_names, fname="pr_curve.png")

Parameters:
 * y_true `numpy.array` corresponding to true labels of validation dataset with shape `[n_samples, ]`
 * scores `numpy.array` corresponding to predicted scores of validation dataset with shape `[n_samples, ]`
 * label_names `list` with items corresponding of label name represented by its index
 * fname `str` path where the image will be saved, works with `.png` format
 
Saves Precision-Recall curve plot for each class on the same image then returns fname

---