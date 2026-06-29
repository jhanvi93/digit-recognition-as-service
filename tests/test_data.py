import numpy as np

from mlops_digit.data import N_CLASSES, N_FEATURES, load_raw, make_splits


def test_load_raw_shape():
    X, y = load_raw()
    assert X.shape[1] == N_FEATURES
    assert X.shape[0] == y.shape[0]
    assert set(np.unique(y)) == set(range(N_CLASSES))


def test_splits_are_disjoint_and_stratified():
    splits = make_splits(test_size=0.2, val_size=0.2, random_state=0)
    total = len(splits.y_train) + len(splits.y_val) + len(splits.y_test)
    X, _ = load_raw()
    assert total == X.shape[0]
    # every class present in each split (stratification)
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert set(np.unique(y)) == set(range(N_CLASSES))


def test_splits_reproducible():
    a = make_splits(random_state=7)
    b = make_splits(random_state=7)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.y_test, b.y_test)
