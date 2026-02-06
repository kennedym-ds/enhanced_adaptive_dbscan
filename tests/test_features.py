"""Tests for DefectFeatureEncoder."""

import numpy as np
import pytest

from wafer_defect_clustering.features import DefectFeatureEncoder
from wafer_defect_clustering.wafer import WaferMap


@pytest.fixture()
def wafer_with_attrs():
    w = WaferMap(diameter_mm=300)
    np.random.seed(42)
    n = 50
    x = np.random.uniform(-100, 100, n)
    y = np.random.uniform(-100, 100, n)
    sizes = np.random.exponential(2.0, n)
    kills = np.random.binomial(1, 0.3, n)
    layers = np.random.choice(['M1', 'M2', 'M3', 'VIA1'], n)
    w.add_defects(x=x, y=y, size=sizes, kill=kills, layer=layers)
    return w


@pytest.fixture()
def spatial_only_wafer():
    w = WaferMap(diameter_mm=300)
    w.add_defects(x=[10, 20, 30], y=[5, 15, 25])
    return w


class TestDefectFeatureEncoder:
    def test_spatial_only(self, spatial_only_wafer):
        enc = DefectFeatureEncoder(size_weight=0, severity_weight=0, layer_weight=0)
        X = enc.fit_transform(spatial_only_wafer)
        assert X.shape == (3, 2)  # only x, y
        assert enc.feature_names == ['x', 'y']

    def test_with_all_attributes(self, wafer_with_attrs):
        enc = DefectFeatureEncoder(
            spatial_weight=1.0,
            size_weight=0.5,
            severity_weight=0.3,
            layer_weight=0.2,
        )
        X = enc.fit_transform(wafer_with_attrs)
        assert X.shape[0] == 50
        assert X.shape[1] == 5  # x, y, size, kill, layer
        assert enc.feature_names == ['x', 'y', 'size', 'kill', 'layer']

    def test_normalisation(self, wafer_with_attrs):
        enc = DefectFeatureEncoder(
            normalize=True, size_weight=0.5, severity_weight=0.3, layer_weight=0.2
        )
        X = enc.fit_transform(wafer_with_attrs)
        # After z-score + weight, columns shouldn't all have same scale
        # but the raw z-score (before weighting) should be ~zero-mean
        assert X.shape[0] == 50

    def test_no_normalisation(self, spatial_only_wafer):
        enc = DefectFeatureEncoder(
            normalize=False, size_weight=0, severity_weight=0, layer_weight=0
        )
        X = enc.fit_transform(spatial_only_wafer)
        # Without normalisation, should match raw coords * weight
        expected_x = np.array([10, 20, 30]) * 1.0
        np.testing.assert_allclose(X[:, 0], expected_x)

    def test_transform_requires_fit(self, spatial_only_wafer):
        enc = DefectFeatureEncoder()
        with pytest.raises(RuntimeError, match='not been fitted'):
            enc.transform(spatial_only_wafer)

    def test_fit_then_transform(self, wafer_with_attrs):
        enc = DefectFeatureEncoder(size_weight=0.5, severity_weight=0.3, layer_weight=0.2)
        _X1 = enc.fit_transform(wafer_with_attrs)

        # Create a new wafer with same attributes structure
        w2 = WaferMap(diameter_mm=300)
        w2.add_defects(x=[10], y=[20], size=[1.5], kill=[0], layer=['M1'])
        X2 = enc.transform(w2)
        assert X2.shape == (1, 5)

    def test_missing_attr_skipped(self, spatial_only_wafer):
        """Encoder doesn't crash if wafer lacks optional attributes."""
        enc = DefectFeatureEncoder(size_weight=0.5, severity_weight=0.3, layer_weight=0.2)
        X = enc.fit_transform(spatial_only_wafer)
        # spatial_only_wafer has no size/kill/layer → should be (3, 2)
        assert X.shape == (3, 2)

    def test_zero_weight_excludes_feature(self, wafer_with_attrs):
        enc = DefectFeatureEncoder(size_weight=0.0, severity_weight=0.3, layer_weight=0.2)
        enc.fit_transform(wafer_with_attrs)
        assert 'size' not in enc.feature_names


class TestPolarFeatures:
    """Tests for polar coordinate feature encoding (r, sin(θ), cos(θ))."""

    def test_polar_weight_zero_no_columns(self, spatial_only_wafer):
        """When polar_weight=0.0 (default), no polar columns are added."""
        enc = DefectFeatureEncoder(
            polar_weight=0.0,
            size_weight=0,
            severity_weight=0,
            layer_weight=0,
        )
        X = enc.fit_transform(spatial_only_wafer)
        assert X.shape == (3, 2)  # only x, y
        assert 'r' not in enc.feature_names
        assert 'sin_theta' not in enc.feature_names
        assert 'cos_theta' not in enc.feature_names

    def test_polar_weight_positive_adds_columns(self, spatial_only_wafer):
        """Positive polar_weight adds r, sin_theta, cos_theta columns."""
        enc = DefectFeatureEncoder(
            polar_weight=0.5,
            size_weight=0,
            severity_weight=0,
            layer_weight=0,
        )
        X = enc.fit_transform(spatial_only_wafer)
        assert X.shape == (3, 5)  # x, y, r, sin_theta, cos_theta
        assert enc.feature_names == ['x', 'y', 'r', 'sin_theta', 'cos_theta']

    def test_polar_feature_values(self, spatial_only_wafer):
        """Polar feature values should match WaferMap.radii and .angles."""
        enc = DefectFeatureEncoder(
            polar_weight=1.0,
            size_weight=0,
            severity_weight=0,
            layer_weight=0,
            normalize=False,
        )
        X = enc.fit_transform(spatial_only_wafer)
        expected_r = spatial_only_wafer.radii
        expected_sin = np.sin(spatial_only_wafer.angles)
        expected_cos = np.cos(spatial_only_wafer.angles)
        # Columns 2, 3, 4 = r, sin_theta, cos_theta (weighted by 1.0)
        np.testing.assert_allclose(X[:, 2], expected_r, atol=1e-10)
        np.testing.assert_allclose(X[:, 3], expected_sin, atol=1e-10)
        np.testing.assert_allclose(X[:, 4], expected_cos, atol=1e-10)

    def test_polar_with_other_attributes(self, wafer_with_attrs):
        """Polar features included alongside other attribute features."""
        enc = DefectFeatureEncoder(
            polar_weight=0.4,
            size_weight=0.5,
            severity_weight=0.3,
            layer_weight=0.2,
        )
        X = enc.fit_transform(wafer_with_attrs)
        names = enc.feature_names
        assert 'r' in names
        assert 'sin_theta' in names
        assert 'cos_theta' in names
        assert 'size' in names
        assert 'kill' in names
        # x, y, r, sin_theta, cos_theta, size, kill, layer = 8
        assert X.shape[1] == 8

    def test_polar_transform_after_fit(self, spatial_only_wafer):
        """Polar features work in transform() after fit_transform()."""
        enc = DefectFeatureEncoder(
            polar_weight=0.5,
            size_weight=0,
            severity_weight=0,
            layer_weight=0,
        )
        enc.fit_transform(spatial_only_wafer)
        w2 = WaferMap(diameter_mm=300)
        w2.add_defects(x=[50], y=[50])
        X2 = enc.transform(w2)
        assert X2.shape == (1, 5)

    def test_get_params_includes_polar(self):
        """polar_weight should be exposed via sklearn get_params."""
        enc = DefectFeatureEncoder(polar_weight=0.7)
        assert enc.polar_weight == 0.7
