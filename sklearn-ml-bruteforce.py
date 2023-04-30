from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import freeze_support
import sys

import traceback
import timeit
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LassoLarsIC, GammaRegressor, TweedieRegressor, BayesianRidge, ARDRegression,  LinearRegression, Ridge, RidgeCV, SGDRegressor, ElasticNet, HuberRegressor, QuantileRegressor, RANSACRegressor, TheilSenRegressor, PoissonRegressor, PassiveAggressiveRegressor, OrthogonalMatchingPursuit
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression, PLSCanonical

def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def initializer(limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))

classifiers = {
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=0),
    "Bagging": BaggingClassifier(n_estimators=10, random_state=0),
    "ExtraTrees (Gini)": ExtraTreesClassifier(criterion="gini", n_estimators=100, random_state=0),
    "ExtraTrees (Entropy)": ExtraTreesClassifier(criterion="entropy", n_estimators=100, random_state=0),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
    "RandomForest": RandomForestClassifier(max_depth=2, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "GaussianProcess": GaussianProcessClassifier(random_state=0),
    "PassiveAggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=0, tol=1e-3),
    "Ridge": RidgeClassifier(),
    "SGDClassifier": make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)),
    "BernoulliNaiveBayes": BernoulliNB(),
    "CategoricalNaiveBayes": CategoricalNB(),
    "ComplementNaiveBayes": ComplementNB(),
    "GaussianNaiveBayes": GaussianNB(),
    "MultinomialNaiveBayes": MultinomialNB(),
    "KNeighbors": KNeighborsClassifier(n_neighbors=3),
    "RadiusNeighbors": RadiusNeighborsClassifier(radius=1.0),
    "NearestCentroid": NearestCentroid(),
    "MLP": MLPClassifier(random_state=1, max_iter=300),
    "LabelPropagation": LabelPropagation(),
    "LinearSVC": make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5)),
    "NuSVC": make_pipeline(StandardScaler(), NuSVC()),
    "SVC": make_pipeline(StandardScaler(), SVC(gamma='auto')),
    "DecisionTree": DecisionTreeClassifier(random_state=0),
    "ExtraTree": ExtraTreeClassifier(random_state=0)
}

def test_classifier(classifier_type, X_train, X_test, y_train, y_test): 
    try:
        start_time = timeit.default_timer()
        print("\nClassifier", classifier_type, "...")
        clf = classifiers[classifier_type]
        clf.fit(X_train, y_train)
        y_predicted = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_predicted)
        print("\nClassifier", classifier_type, "accuracy is", accuracy)
        stop_time = timeit.default_timer()
        print("Classifier", classifier_type, "completed in", format_time(stop_time-start_time))
        return [classifier_type, accuracy]
    except Exception: 
        print("\nError for Classifier", classifier_type)
        traceback.print_exc()

def test_classifiers(X_train, X_test, y_train, y_test):
    result_queue = []
    if (LINUX):
        with ProcessPool(max_workers=POOL,initializer=initializer, initargs=(MAX_MEMORY,)) as pool:             
            multiple_results = [(pool.schedule(test_classifier, args=(key, X_train, X_test, y_train, y_test), timeout=TIMEOUT_SECONDS), key) for key in classifiers]
            for res in multiple_results:
                try:
                    tmp = res[0].result()
                    if tmp is not None:
                        result_queue.append(tmp)
                except TimeoutError:
                    print("\nClassifier", res[1], "exceeded the time limit.")
                except MemoryError:
                    print("\nClassifier", res[1], "exceeded the memory limit.")
    else: 
        with ProcessPool(max_workers=POOL) as pool:             
            multiple_results = [(pool.schedule(test_classifier, args=(key, X_train, X_test, y_train, y_test), timeout=TIMEOUT_SECONDS), key) for key in classifiers]
            for res in multiple_results:
                try:
                    tmp = res[0].result()
                    if tmp is not None:
                        result_queue.append(tmp)
                except TimeoutError:
                    print("\nClassifier", res[1], "exceeded the time limit.")

    accuracy = {}
    for value in result_queue:
        accuracy[value[0]] = value[1]
    accuracy = {k: v for k, v in sorted(accuracy.items(), key=lambda item: item[1], reverse=True)}
    
    print("Results: \n")
    untested = set(classifiers.keys())
    i = 1
    for key in accuracy:
        print(i, key, accuracy[key])
        untested.remove(key)
        i += 1
    print("\nUntested Classifiers:", untested)

regressors = {
    "AdaBoost (square)" : AdaBoostRegressor(random_state=0, n_estimators=100, loss="square"),
    "AdaBoost (linear)" : AdaBoostRegressor(random_state=0, n_estimators=100, loss="linear"),
    "Adaboost (exponential)" : AdaBoostRegressor(random_state=0, n_estimators=100, loss="exponential"),
    "Bagging" : BaggingRegressor(n_estimators=10, random_state=0),
    "Bagging (svr)": BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0),
    "ExtraTrees (abs err)" : ExtraTreesRegressor(criterion = "absolute_error", n_estimators=100, random_state=0),
    "ExtraTrees (sq err)" : ExtraTreesRegressor(criterion = "squared_error", n_estimators=100, random_state=0),
    "GradientBoosting (huber)" : GradientBoostingRegressor(random_state=0,loss="huber"),
    "GradientBoosting (sq err)" : GradientBoostingRegressor(random_state=0,loss="squared_error"),
    "GradientBoosting (abs err)" : GradientBoostingRegressor(random_state=0,loss="absolute_error"),
    "Random Forest (sq err)" : RandomForestRegressor(max_depth=2, random_state=0,criterion="squared_error"),
    "Random Forst (abs err)" : RandomForestRegressor(max_depth=2, random_state=0,criterion="absolute_error"),
    "Random Forest (poisson)" : RandomForestRegressor(max_depth=2, random_state=0,criterion="poisson"),
    "HistGradientBoosting (sq err)" : HistGradientBoostingRegressor(loss="squared_error"),
    "HistGradientBoosting (abs err)" : HistGradientBoostingRegressor(loss="absolute_error"),
    "HistGradientBoosting (poisson)" : HistGradientBoostingRegressor(loss="poisson"),
    "GaussianProcess" : GaussianProcessRegressor(random_state=0),
    "Linear" : LinearRegression(),
    "Ridge (Linear)" : Ridge(),
    "RidgeCV" : RidgeCV(),
    "SGDRegressor (elasticnet)" : make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3,penalty="elasticnet")),
    "SGDRegressor (l2)" : make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3,penalty="l2")),
    "SGDRegressor (l1)" : make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3,penalty="l1")),
    "Elastic Net (random)" : ElasticNet(random_state=0,selection="random"),
    "Elastic Net (cyclic)" : ElasticNet(random_state=0,selection="cyclic"),
    "ARD" : ARDRegression(),
    "BayesianRidge" : BayesianRidge(),
    "Huber" : HuberRegressor(),
    "Quantile (highs-ds)" : QuantileRegressor(quantile=0.8, solver="highs-ds"),
    "Quantile (highs-ipm)" : QuantileRegressor(quantile=0.8, solver="highs-ipm"),
    "Quantile (highs)" : QuantileRegressor(quantile=0.8, solver="highs"),
    "Quantile (interior-point)" : QuantileRegressor(quantile=0.8, solver="interior-point"),
    "Quantile (revised simplex)" : QuantileRegressor(quantile=0.8, solver="revised simplex"),
    "RANSAC": RANSACRegressor(random_state=0),
    "TheilSenRegressor" : TheilSenRegressor(random_state=0),
    "PoissonRegressor" : PoissonRegressor(),
    "TweedieRegressor (auto)" : TweedieRegressor(link="auto"),
    "TweedieRegressor (identity)" : TweedieRegressor(link="identity"),
    "TweedieRegressor (log)" : TweedieRegressor(link="log"),
    "GammaRegressor" : GammaRegressor(),
    "PassiveAggressiveRegressor (epsilon_insensitive)" :  PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3, loss="epsilon_insensitive"),
    "PassiveAggressiveRegressor (squared_epsilon_insensitive)" :  PassiveAggressiveRegressor(max_iter=100, random_state=0, tol=1e-3, loss="squared_epsilon_insensitive"),
    "KNeighbors" : KNeighborsRegressor(n_neighbors=3),
    "Radius Neighbors" : RadiusNeighborsRegressor(radius=1.0),
    "MLP" : MLPRegressor(random_state=1, max_iter=500),
    "DecisionTree" : DecisionTreeRegressor(random_state=0),
    "Extra Tree" : ExtraTreeRegressor(random_state=0),
    "Kernel Ridge" : KernelRidge(alpha=1.0),
    "Linear SVR (epsilon_insensitive)" : make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5, loss="epsilon_insensitive")),
    "Linear SVR (squared_epsilon_insensitive)" : make_pipeline(StandardScaler(), LinearSVR(random_state=0, tol=1e-5, loss="squared_epsilon_insensitive")),
    "nuSVR" : make_pipeline(StandardScaler(), NuSVR(C=1.0, nu=0.1)),
    "SVR" : make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
    "LassoLarsIC (bic)" : LassoLarsIC(criterion='bic', normalize=False),
    "LassoLarsIC (aic)" : LassoLarsIC(criterion='aic', normalize=False),
    "PLS" : PLSRegression(n_components=2),
    "OrthogonalMatchingPursuit" : OrthogonalMatchingPursuit(),
    "PLSCanonical" : PLSCanonical(n_components=2)
}

def test_regressor(regressor_type, X_train, X_test, y_train, y_test): 
    try:
        start_time = timeit.default_timer()
        print("\nRegressor", regressor_type, "...")
        reg = regressors[regressor_type]
        reg.fit(X_train, y_train)
        y_predicted = reg.predict(X_test)
        accuracy = metrics.mean_absolute_error(y_test, y_predicted)
        print("\nRegressor", regressor_type, "accuracy is", accuracy)
        stop_time = timeit.default_timer()
        print("Regressor", regressor_type, "completed in", format_time(stop_time-start_time))
        return [regressor_type, accuracy]
    except Exception: 
        print("\nError for Regressor", regressor_type)
        traceback.print_exc()

def test_regressors(X_train, X_test, y_train, y_test):
    result_queue = []
    if (LINUX):
        with ProcessPool(max_workers=POOL,initializer=initializer, initargs=(MAX_MEMORY,)) as pool:             
            multiple_results = [(pool.schedule(test_regressor, args=(key, X_train, X_test, y_train, y_test), timeout=TIMEOUT_SECONDS), key) for key in regressors]
            for res in multiple_results:
                try:
                    tmp = res[0].result()
                    if tmp is not None:
                        result_queue.append(tmp)
                except TimeoutError:
                    print("\nClassifier", res[1], "exceeded the time limit.")
                except MemoryError:
                    print("\nClassifier", res[1], "exceeded the memory limit.")
    else: 
        with ProcessPool(max_workers=POOL) as pool:             
            multiple_results = [(pool.schedule(test_regressor, args=(key, X_train, X_test, y_train, y_test), timeout=TIMEOUT_SECONDS), key) for key in regressors]
            for res in multiple_results:
                try:
                    tmp = res[0].result()
                    if tmp is not None:
                        result_queue.append(tmp)
                except TimeoutError:
                    print("\nClassifier", res[1], "exceeded the time limit.") 

    mae = {}
    for value in result_queue:
        mae[value[0]] = value[1]
    mae = {k: v for k, v in sorted(mae.items(), key=lambda item: item[1], reverse=False)}

    print("Results: \n")
    untested = set(regressors.keys())
    i = 1
    for key in mae:
        print(i, key, mae[key])
        untested.remove(key)
        i += 1
    print("\nUntested Regressors:", untested)

if __name__ == "__main__":
    TIMEOUT_SECONDS = 60 * 10
    POOL = 4
    MAX_MEMORY = 1572864
    SILENT = False

    if ("win" in sys.platform):
        print("Windows Detected")
        freeze_support()
        LINUX = False
    else: 
        print("Linux Assumed:")
        import resource
        LINUX = True
    if SILENT:
        import warnings
        warnings.filterwarnings("ignore")