import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def is_categorical(df, feature):
    unique_values = df[feature].unique()
    if len(unique_values) != 2:
        return False
    if (np.sort(unique_values) != np.array([0, 1])).all():
        return False
    return True


def get_percentage_label_not(df,
                             feature,
                             label,
                             label_column):

    positive_percentage_label = (df[df[feature] == 1][label_column] == label).mean().item()
    positive_percentage_not_label = (df[df[feature] != 1][label_column] == label).mean().item()

    return {
        f"{label}": {
            'positive': positive_percentage_label,
            'negative': 1 - positive_percentage_label
        },
        f"not_{label}": {
            'positive': positive_percentage_not_label,
            'negative': 1 - positive_percentage_not_label
        },
    }


def get_all_categoric_percentages(df,
                                  features,
                                  label,
                                  label_column):
    data = {}
    for feature in features:
        feature_info = get_percentage_label_not(df,
                                                feature,
                                                label,
                                                label_column)

        data[feature] = feature_info

    return data


def save_per_feature_analysis(keys,
                              predicted_label_name,
                              predicted_label_names,
                              label_idx,
                              prediction_info,
                              label_column,
                              df):
    predicted_label_list = [f"{predicted_label_name}", f"non_{predicted_label_name}"]
    fig, axes = plt.subplots(len(keys), 2, figsize=(20, len(keys) * 5))

    for key_idx in range(len(keys)):

        key = keys[key_idx]

        if is_categorical(df, key):
            feature_info = prediction_info[key]

            keylist = list(feature_info.keys())

            info_feature = feature_info[keylist[0]]
            info_non_feature = feature_info[keylist[1]]

            axes[key_idx][0].title.set_text(f"Has feature {key}")
            axes[key_idx][1].title.set_text(f"Does not have feature {key}")

            axes[key_idx][0].pie([info_feature["positive"], info_feature["negative"]],
                                 labels=predicted_label_list,
                                 autopct='%1.1f%%',
                                 shadow=True,
                                 startangle=90)

            axes[key_idx][1].pie([info_non_feature["positive"], info_non_feature["negative"]],
                                 labels=predicted_label_list,
                                 autopct='%1.1f%%',
                                 shadow=True,
                                 startangle=90)

        else:
            axes[key_idx][0].title.set_text(f"{key} distribution for {predicted_label_name}")
            sns.distplot(np.array(df[df[label_column] == label_idx][key]),
                         ax=axes[key_idx][0])


            axes[key_idx][1].title.set_text(f"{key} distribution for all labels")

            for j in range(len(predicted_label_names)):
                sns.distplot(np.array(df[df[label_column] != j][key]),
                             ax=axes[key_idx][1],
                             label=predicted_label_names[j])

            axes[key_idx][1].legend()
    plt.savefig(f"{predicted_label_name}_per_feature_analysis.png")
    return f"{predicted_label_name}_per_feature_analysis.png"


def eval_classification_model_predictions_per_feature(df_filename,
                                                      classifier,
                                                      label_column_name,
                                                      label_names,
                                                      to_drop,
                                                      preprocess=lambda i: i,
                                                      use_torch=False,
                                                      device=None):

    df = pd.read_csv(df_filename)
    df = preprocess(df)
    df = df.drop([label_column_name] + to_drop, axis=1)
    arr = np.array(df)

    if use_torch:
        import torch
        preds = classifier(torch.Tensor(arr).to(device)).argmax(axis=1)
        df['prediction'] = preds.cpu().numpy()
    else:
        preds = classifier.predict(arr)
        df['prediction'] = preds

    features = df.columns.tolist()[:-1]
    categoric_features = list(filter(lambda i: is_categorical(df, i), features))
    analysis_filenames = []

    for i in range(len(label_names)):
        prediction_info = get_all_categoric_percentages(df, categoric_features, i, 'prediction')
        predicted_label_name = label_names[i]

        analysis_filenames.append(save_per_feature_analysis(features,
                                                            predicted_label_name,
                                                            label_names,
                                                            i,
                                                            prediction_info,
                                                            'prediction',
                                                            df))
        plt.clf()
    return analysis_filenames
