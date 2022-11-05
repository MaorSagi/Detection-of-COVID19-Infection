import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, classification_report
from imblearn.over_sampling import SVMSMOTE
from joblib import dump, load
from consts import filtered_features, feature_dict

METRICS = ["Accuracy", "F1", "Sensitivity", "Specificity", "AUROC"]
METRICS_NUM = len(METRICS)


def preprocessing(df):
    # preprocessing 5.2

    df_filtered_features = df[filtered_features]
    df_drop_nan = df_filtered_features.dropna(axis=0, thresh=0.05 * len(df_filtered_features.columns))
    # print("Missing ratio: ", df_drop_nan.isnull().sum() * 100 / len(df_drop_nan))
    df_drop_nan = df_drop_nan.rename(columns=feature_dict, inplace=False)
    X = df_drop_nan.drop(columns='SARS-Cov-2 exam result')
    y = df_drop_nan['SARS-Cov-2 exam result'].map({'positive': 1, 'negative': 0}).astype(int)
    imputer = IterativeImputer(random_state=42)
    imputer.fit(X.values)
    X = pd.DataFrame(imputer.transform(X.values), columns=X.columns)
    # print("Missing ratio: ", X.isnull().sum() * 100 / len(X))
    return X, y


def hyperparameter_tuning(X, y, writing_to_results_file=False):
    # SMOTE re-sampling
    X, y = SVMSMOTE(random_state=42).fit_resample(X, y)
    models = grid_search_prep()
    for model_name in models:
        best_score_for_model = 0
        best_model_for_model = None
        model = models[model_name]
        if model['load']:
            models[model_name]['model'] = load(model_name + '.joblib')
            continue
        inner_cv = KFold(shuffle=True, random_state=42)
        outer_cv = KFold(shuffle=True, random_state=42)
        clf = GridSearchCV(estimator=model['model'], param_grid=model['params'],
                           cv=inner_cv, scoring='f1',
                           n_jobs=-1)
        nested_score = cross_validate(clf, X=X, y=y, cv=outer_cv,
                                      scoring='f1',
                                      n_jobs=-1,
                                      return_estimator=True)
        for estimator in nested_score['estimator']:
            if estimator.best_score_ > best_score_for_model:
                best_score_for_model = estimator.best_score_
                best_model_for_model = estimator.best_estimator_
        dump(best_model_for_model, model_name + ', SCORE=' + str(best_score_for_model) + '.joblib')
        models[model_name]['model'] = best_model_for_model
        if writing_to_results_file:
            with open("results.txt", "a") as results_file:
                results_file.write(model_name.upper() + ' Model\nBest Model Parameters:' + str(
                    best_model_for_model.get_params()) + '\n')

    return models


def training_model(model, X, y):
    NUM_ITERATIONS = 10
    final_scores = dict(Accuracy=[], F1=[], Sensitivity=[], Specificity=[], AUROC=[])
    for i in range(NUM_ITERATIONS):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=i)
        # SMOTE re-sampling - only on train set
        X_train, y_train = SVMSMOTE(random_state=42).fit_resample(X_train, y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results = classification_report(y_test, y_pred, output_dict=True)
        final_scores['Accuracy'].append(results['accuracy'])
        final_scores['F1'].append(results['macro avg']['f1-score'])
        final_scores['Sensitivity'].append(results['1']['recall'])
        final_scores['Specificity'].append(results['0']['recall'])
        final_scores['AUROC'].append(roc_auc_score(y_test, y_pred))
    for score in final_scores:
        score_list = final_scores[score]
        final_scores[score] = dict(mean=np.mean(np.array(score_list)),
                                   std=np.std(np.array(score_list)))
    return final_scores


def grid_search_prep():
    lr_grid_param = dict()
    rf_grid_param = dict(n_estimators=[10, 20, 30] + [x for x in np.arange(45, 105, 5)],
                         max_depth=[4, 8, 16, 32, 64])
    xgb_grid_param = dict(n_estimators=[10, 20, 30] + [x for x in np.arange(45, 105, 5)],
                          max_depth=[4, 8, 16, 32, 64], learning_rate=[0.1, 0.05, 0.01])
    cb_grid_param = dict(n_estimators=[10, 20, 30] + [x for x in np.arange(45, 105, 5)],
                         learning_rate=[0.1, 0.05, 0.01])  # max_depth=[4, 8, 16, 32, 64]
    lgbm_grid_param = dict(n_estimators=[10, 20, 30] + [x for x in np.arange(45, 105, 5)],
                           max_depth=[4, 8, 16, 32, 64], learning_rate=[0.1, 0.05, 0.01])
    lr_clf = LogisticRegression(random_state=42)
    rf_clf = RandomForestClassifier(random_state=42)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)  # use_label_encoder=False,
    cb_clf = CatBoostClassifier(random_state=42)
    lgbm_clf = LGBMClassifier(random_state=42)
    models = dict(lr=dict(model=lr_clf, params=lr_grid_param, load=True),
                  rf=dict(model=rf_clf, params=rf_grid_param, load=True),
                  xgb=dict(model=xgb_clf, params=xgb_grid_param, load=True),
                  cb=dict(model=cb_clf, params=cb_grid_param, load=True),
                  lgbm=dict(model=lgbm_clf, params=lgbm_grid_param, load=True),
                  )
    return models


def get_X_after_features_extraction(X, y, degree):
    pf = PolynomialFeatures(degree=degree).fit(X)
    X_fe = pd.DataFrame(pf.transform(X), columns=pf.get_feature_names(X.columns))
    X_fe = X_fe.drop(columns=X.columns)
    skb = SelectKBest(mutual_info_classif, k=5).fit(X_fe, y)
    selected = skb.get_support(indices=True)
    X_fe_new = X_fe.iloc[:, selected]
    X = pd.concat([X, X_fe_new], axis=1)
    return X


def fit_predict(models, X, y, scores_dict=None, writing_to_results_file=False, with_shap=False):
    for model_name in models:
        if writing_to_results_file:
            with open("results.txt", "a") as results_file:
                results_file.write(model_name.upper() + '\n')
        model = models[model_name]['model']
        scores = training_model(model, np.array(X), np.array(y))
        scores_sum = 0
        for score in scores:
            scores_sum += scores[score]['mean']
            if scores_dict:
                scores_dict[score][model_name].append(scores[score]['mean'])
            if writing_to_results_file:
                with open("results.txt", "a") as results_file:
                    results_file.write(
                        score + ": " + str(scores[score]['mean']) + "±" + str(scores[score]['std']) + '\n')
        if scores_dict:
            scores_dict['Mean Scores'][model_name].append(scores_sum / METRICS_NUM)
        if with_shap:
            if model_name == 'lr':
                continue
            explainer = shap.Explainer(model)
            shap_values = explainer(X)
            shap.summary_plot(shap_values[:, :, 0] if len(shap_values.shape) > 2 else shap_values)
            shap.summary_plot(shap_values[:, :, 0] if len(shap_values.shape) > 2 else shap_values, plot_type='bar')
    return scores_dict


if __name__ == '__main__':
    feature_extraction_test = False
    train_with_feature_extraction = True
    BEST_DEGREE = 5
    df = pd.read_excel("dataset.xlsx", engine="openpyxl")
    X, y = preprocessing(df)
    if feature_extraction_test:
        metrics_with_mean = METRICS + ["Mean Scores"]
        scores_dict_with = {key: dict(lr=[], rf=[], xgb=[], cb=[], lgbm=[]) for key in metrics_with_mean}
        scores_dict_without = {key: dict(lr=[], rf=[], xgb=[], cb=[], lgbm=[]) for key in metrics_with_mean}
        DEGREES = 7
        for n in range(2, DEGREES):
            try:
                X_new = get_X_after_features_extraction(X, y, n)
                models = hyperparameter_tuning(np.array(X_new), np.array(y))
                scores_dict_with = fit_predict(models, X_new, y, scores_dict=scores_dict_with)
            except Exception as err:
                print("Degree=", n, ' Error: ', err)
        x_degrees = np.arange(2, DEGREES)
        models = hyperparameter_tuning(np.array(X), np.array(y))
        scores_dict_without = fit_predict(models, X, y, scores_dict=scores_dict_without)
        for metric in metrics_with_mean:
            for model_name in models:
                x_model = x_degrees
                scores_dict_without[metric][model_name] = scores_dict_without[metric][model_name] * len(x_degrees)
                y_model = [100 * (item1 - item2) for (item1, item2) in
                           zip(scores_dict_with[metric][model_name], scores_dict_without[metric][model_name])]
                plt.plot(x_model, y_model, label=model_name.upper())

            plt.xlabel('Degrees')
            plt.ylabel(metric + ' * 100')
            plt.title('Degrees - ' + metric + ' Difference Graph')
            plt.xticks(x_degrees)
            plt.legend()
            plt.show()

    else:
        if train_with_feature_extraction:
            X = get_X_after_features_extraction(X, y, BEST_DEGREE)
        models = hyperparameter_tuning(np.array(X), np.array(y))
        fit_predict(models, X, y, with_shap=True, writing_to_results_file=True)
