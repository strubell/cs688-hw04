cs688-hw04
==========

Restricted Boltzmann machine for the MNIST handwritten digits dataset.

* All of the below assume models in a directory one level up called "models", and data in a directory one level up called "data".

* Generated data and models (e.g. for SVMlight) are also placed in these directories, and will fail if they do not exist.

* Directory names can be changed in `load_data.py`.

* Actual RBM implementation contained in `rbm.py`.

Question 2: Inference, Burn-in and Autocorrelation
----------

To get the plots reported in question 2:

```
$ python q2.py
```

Question 3: Learning
----------

To get the plots reported in question 3:

```
$ python q3.py
```

Question 4: Feature Extraction and Classification
----------

To generate the labeled feature vectors for embeddings and raw data in SVMlight format:

```
$ python q4.py
```

Bash scripts are included for training/testing a a multiclass SVM classifier using SVMlight on the above genrated data.