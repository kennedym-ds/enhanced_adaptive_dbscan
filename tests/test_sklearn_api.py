import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN


def make_blob(n=100, seed=0):
    rng = np.random.default_rng(seed)
    # two clusters around (-1, -1) and (1, 1)
    a = rng.normal(loc=-1.0, scale=0.2, size=(n//2, 2))
    b = rng.normal(loc=1.0, scale=0.2, size=(n - n//2, 2))
    return np.vstack([a, b])


def test_fit_predict_returns_labels_shape():
    X = make_blob(60, seed=1)
    est = EnhancedAdaptiveDBSCAN(k=10, n_jobs=1)
    labels = est.fit_predict(X)
    assert labels.shape[0] == X.shape[0]
    assert hasattr(est, 'labels_')


def test_plot_raises_before_fit():
    X = make_blob(30, seed=2)
    est = EnhancedAdaptiveDBSCAN(k=5, n_jobs=1)
    with pytest.raises(NotFittedError):
        est.plot_clusters(X)


def test_evaluate_raises_before_fit():
    X = make_blob(30, seed=3)
    est = EnhancedAdaptiveDBSCAN(k=5, n_jobs=1)
    with pytest.raises(NotFittedError):
        # labels arg is ignored for fitted check
        est.evaluate_clustering(X, np.full(X.shape[0], -1))
