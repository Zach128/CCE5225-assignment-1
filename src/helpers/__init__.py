import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def fitGridSearch(search: GridSearchCV, X_train, y_train):
    """
    Perform a fitting on a preconfigured grid-search.
    Once done, report the results

    Args:
        search (GridSearchCV): A valid, configured grid search object.
        X_train (Iterable): Your models training data.
        y_train (Iterable): Your training datas classifications.

    Yields:
        search (GridSearchCV): The same grid search used.
    """

    search.fit(X_train, y_train)

    # Print the results of the above grid search.
    print('Best Score: %s' % search.best_score_)
    print('Best Hyperparameters: %s\n' % search.best_params_)

    # Report all the scores for all the various combinations.
    means = search.cv_results_['mean_test_score']
    stds = search.cv_results_['std_test_score']

    # Print the mean accuracy and expected deviation for all parameter combinations.
    print('Performance breakdown for each fit:')
    for mean, std, params in zip(means, stds, search.cv_results_['params']):
        print('%0.3f (+/-%0.03f) for %r' % (mean, std * 2, params))
    
    return search

def testAlgorithm(search: GridSearchCV, X_test, y_test):
    """
    Test a fitted GridSearch result, reporting on the performance and time duration.

    Args:
        search (GridSearchCV): A valid, configured grid search object.
        X_test (Iterable): Your models test data.
        y_test (Iterable): Your test datas classifications.

    Yields:
        search (GridSearchCV): The same grid search used.
    """

    # Test the selected hyperparameters on the test set.
    timeStart = time.time() * 1000
    predictions = search.predict(X_test)
    timeEnd = time.time() * 1000
    totalTime = timeEnd - timeStart

    print('\nTotal prediction time: {}ms\nAverage time/prediction: {}ms'.format(totalTime, totalTime / len(X_test)))

    print('\nConfusion matrix:')
    print(confusion_matrix(y_test, predictions))

    print('\nClassification report:')
    print(classification_report(y_test, predictions))
