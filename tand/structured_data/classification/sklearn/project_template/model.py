from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

models = {
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "ada_boost": AdaBoostClassifier,
    "naive_bayes": GaussianNB,
    "logistic_regression": LogisticRegression

}


def parse_model_option(model_name, kwargs):
    global models
    return models[model_name](**kwargs)
