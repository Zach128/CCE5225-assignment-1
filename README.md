## Dataset breakdown:

* 1st line:
  * 1st number: Signal events in file.
  * 2nd number: Background events in file.
* Each line past line 1:
  * 1 event (either signal or background):
    * 50 variables known as Particle IDentification variables.
  * Events are 1 of 2 classes - signal, or background.
* Aim is to classify events correctly into the two classes.

## Setup

1. Download [MiniBooNE dataset](https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification).

## Goals

Implement the below classifiers, ensuring each individual classifier meets the shared objectives below.

### Classifier implementations

* [x] Vanilla neural network
* [x] Support Vector Machine
* [ ] Random forest

### Shared Objectives

* [x] Load and transform the dataset into a feature matrix. Expected format of (x: event, y: feature).
* [x] [Scale](https://en.wikipedia.org/wiki/Feature_scaling) the input features to a discreet value range.
* [x] Re-shuffle the dataset, dividing into a training and test set of 80% and 20% respectively.
* nn: [x] svm: [x] rf: [x] Identify the hyperparameters which result in the best accuracy. Do so using a [grid-search](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/) and test it using [5-fold cross-validation](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/) (Hence the 80% 20% partitioning above.)
  * nn: [ ] svm: [ ] rf: [ ] Comment/provide feedback on the performance influence of different hyperparameter values per model.
* nn: [x] svm: [x] rf: [x] Report the following data:
  * nn: [x] svm: [x] rf: [x] Training time
  * nn: [x] svm: [x] rf: [x] Tested hyperparameters
  * nn: [x] svm: [x] rf: [x] [Confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) of the per-class accuracies achieved by running on the unseen test set.
* nn: [ ] svm: [ ] rf: [ ] Evaluate each model's performance, commenting on the performance and providing an explanation with reasons on why - in your opinion - the highest-performing model gave the best results (see [Information Theory](https://en.wikipedia.org/wiki/Information_theory)).
