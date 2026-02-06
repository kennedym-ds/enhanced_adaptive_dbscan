"""Tests for WaferMap, WaferGeometry, and ZoneDefinition."""

import numpy as np
import pandas as pd
import pytest

from wafer_defect_clustering.wafer import (
    STANDARD_ZONES,
    WaferGeometry,
    WaferMap,
)


class TestWaferGeometry:
    def test_defaults(self):
        g = WaferGeometry()
        assert g.diameter_mm == 300.0
        assert g.radius_mm == 150.0
        assert g.edge_exclusion_mm == 3.0
        assert g.usable_radius_mm == 147.0
        assert g.shape == 'circular'

    def test_custom(self):
        g = WaferGeometry(diameter_mm=200, edge_exclusion_mm=2, flat_length_mm=57.5)
        assert g.radius_mm == 100.0
        assert g.flat_length_mm == 57.5

    def test_notch_angle_rad(self):
        g = WaferGeometry(notch_angle_deg=270)
        assert np.isclose(g.notch_angle_rad, 3 * np.pi / 2)


class TestWaferMap:
    def test_empty_wafer(self):
        w = WaferMap()
        assert w.n_defects == 0
        assert w.coordinates.shape == (0, 2)
        assert w.radii.shape == (0,)
        assert w.angles.shape == (0,)

    def test_add_defects_basic(self):
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[10, -20, 50], y=[5, 30, -10])
        assert w.n_defects == 3
        assert w.coordinates.shape == (3, 2)
        np.testing.assert_allclose(w.coordinates[0], [10, 5])

    def test_add_defects_with_attributes(self):
        w = WaferMap()
        w.add_defects(x=[1, 2], y=[3, 4], size=[0.5, 1.2], kill=[1, 0])
        assert 'size' in w.attribute_names
        assert 'kill' in w.attribute_names
        np.testing.assert_allclose(w.get_attribute('size'), [0.5, 1.2])

    def test_add_defects_incremental(self):
        w = WaferMap()
        w.add_defects(x=[1, 2], y=[3, 4])
        w.add_defects(x=[5], y=[6])
        assert w.n_defects == 3

    def test_add_defects_shape_mismatch_raises(self):
        w = WaferMap()
        with pytest.raises(ValueError, match='same shape'):
            w.add_defects(x=[1, 2], y=[3])

    def test_add_defects_attr_length_mismatch_raises(self):
        w = WaferMap()
        with pytest.raises(ValueError, match='length'):
            w.add_defects(x=[1, 2], y=[3, 4], size=[0.5])

    def test_get_unknown_attribute_raises(self):
        w = WaferMap()
        w.add_defects(x=[1], y=[2])
        with pytest.raises(KeyError, match='nope'):
            w.get_attribute('nope')

    def test_radii(self):
        w = WaferMap()
        w.add_defects(x=[3, 0], y=[4, 5])
        np.testing.assert_allclose(w.radii, [5.0, 5.0])

    def test_distance_to_edge_circular(self):
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[0, 140], y=[0, 0])  # centre and near-edge
        dte = w.distance_to_edge()
        assert dte[0] == 150.0  # centre
        assert dte[1] == 10.0  # 140mm from centre on 150mm radius

    def test_distance_to_edge_square(self):
        w = WaferMap(diameter_mm=200, shape='square')
        w.add_defects(x=[0, 90], y=[0, 0])
        dte = w.distance_to_edge()
        assert dte[0] == 100.0
        assert dte[1] == 10.0

    def test_is_inside_circular(self):
        w = WaferMap(diameter_mm=200)
        w.add_defects(x=[0, 50, 120], y=[0, 50, 0])
        inside = w.is_inside_wafer()
        assert inside[0] is np.True_
        assert inside[1] is np.True_
        assert inside[2] is np.False_  # > 100mm radius

    def test_is_inside_square(self):
        w = WaferMap(diameter_mm=200, shape='square')
        w.add_defects(x=[0, 110], y=[0, 0])
        inside = w.is_inside_wafer()
        assert inside[0]
        assert not inside[1]

    def test_is_inside_external_coords(self):
        w = WaferMap(diameter_mm=200)
        inside = w.is_inside_wafer(x=[0, 200], y=[0, 0])
        assert inside[0]
        assert not inside[1]


class TestZones:
    @pytest.fixture()
    def wafer_with_zones(self):
        w = WaferMap(diameter_mm=300)
        # Centre point, middle point, edge point
        w.add_defects(x=[10, 70, 140], y=[0, 0, 0])
        return w

    def test_standard_zone_names(self):
        names = [z.name for z in STANDARD_ZONES]
        assert names == ['center', 'middle', 'edge']

    def test_get_zone_mask_center(self, wafer_with_zones):
        mask = wafer_with_zones.get_zone_mask('center')
        assert mask[0]  # 10mm from centre = center zone
        assert not mask[1]  # 70mm = middle
        assert not mask[2]  # 140mm = edge

    def test_get_zone_mask_edge(self, wafer_with_zones):
        mask = wafer_with_zones.get_zone_mask('edge')
        assert not mask[0]
        assert not mask[1]
        assert mask[2]

    def test_get_zone_mask_edge_custom_ring(self, wafer_with_zones):
        mask = wafer_with_zones.get_zone_mask('edge', ring_width_mm=15)
        # 140mm → distance to edge = 10mm < 15mm → True
        assert mask[2]
        # 70mm → distance to edge = 80mm > 15mm → False
        assert not mask[1]

    def test_get_zone_label(self, wafer_with_zones):
        labels = wafer_with_zones.get_zone_label()
        assert labels[0] == 'center'
        assert labels[1] == 'middle'
        assert labels[2] == 'edge'

    def test_unknown_zone_raises(self, wafer_with_zones):
        with pytest.raises(ValueError, match='Unknown zone'):
            wafer_with_zones.get_zone_mask('neptune')


class TestDieMap:
    def test_basic_die_aggregation(self):
        w = WaferMap(diameter_mm=300)
        # 3 defects in same die, 1 in another
        w.add_defects(x=[1, 2, 3, 15], y=[1, 2, 3, 15])
        df = w.to_die_map(die_size_mm=(10, 10))
        assert isinstance(df, pd.DataFrame)
        assert 'count' in df.columns
        assert df['count'].sum() == 4

    def test_empty_wafer_die_map(self):
        w = WaferMap()
        df = w.to_die_map()
        assert len(df) == 0


class TestConversions:
    def test_to_dataframe(self):
        w = WaferMap()
        w.add_defects(x=[1, 2], y=[3, 4], size=[0.5, 1.0])
        df = w.to_dataframe()
        assert list(df.columns) == ['x', 'y', 'size']
        assert len(df) == 2

    def test_repr(self):
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[1], y=[2])
        assert '300' in repr(w)
        assert 'n_defects=1' in repr(w)
