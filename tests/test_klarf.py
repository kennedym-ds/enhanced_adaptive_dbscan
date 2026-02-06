"""Tests for wafer_defect_clustering.klarf — KLARF file ingestion."""

from __future__ import annotations

import os
import tempfile
from textwrap import dedent

import pytest

from wafer_defect_clustering.klarf import load_klarf, parse_klarf_string
from wafer_defect_clustering.wafer import WaferMap

# ---------------------------------------------------------------------------
# Synthetic KLARF content (minimal v1.x format)
# ---------------------------------------------------------------------------

SAMPLE_KLARF = dedent("""\
FileVersion 1 2;
FileTimestamp 2026-01-15 10:30:00;
InspectionStationID "KLA" "2139" "1";
SampleType WAFER;
ResultTimestamp 2026-01-15 10:30:00;
LotID "LOT001";
SampleSize 1 300000;
DeviceID "DEVICE_A";
SetupID "SETUP1" 2026-01-15 10:00:00;
StepID "STEP1";
WaferID "W01";
Slot 1;
SampleOrientationMarkType NOTCH;
OrientationMarkLocation DOWN;
DiePitch 10000 10000;
DieOrigin 0 0;
SampleCenterLocation 150000 150000;
InspectionTest 1;
SampleTestPlan 2 0 100000 0 150000;
AreaPerTest 6000000 5000000;
DefectRecordSpec 6 DEFECTID XREL YREL DEFECTSIZE CLASSNUMBER ROUGHBINNUMBER;
DefectList
1 50000 30000 150 1 1;
2 -20000 80000 230 2 1;
3 120000 -10000 90 1 2;
4 0 0 500 3 1;
5 -100000 -50000 180 2 2;
EndOfFile;
""")

SAMPLE_KLARF_TWO_WAFERS = dedent("""\
FileVersion 1 2;
FileTimestamp 2026-01-15 10:30:00;
InspectionStationID "KLA" "2139" "1";
SampleType WAFER;
LotID "LOT001";
SampleSize 1 300000;
WaferID "W01";
Slot 1;
SampleOrientationMarkType NOTCH;
OrientationMarkLocation DOWN;
DiePitch 10000 10000;
DieOrigin 0 0;
SampleCenterLocation 150000 150000;
DefectRecordSpec 4 DEFECTID XREL YREL DEFECTSIZE;
DefectList
1 10000 20000 100;
2 30000 40000 200;
EndOfFile;
WaferID "W02";
Slot 2;
SampleOrientationMarkType NOTCH;
OrientationMarkLocation DOWN;
DiePitch 10000 10000;
DieOrigin 0 0;
SampleCenterLocation 150000 150000;
DefectRecordSpec 4 DEFECTID XREL YREL DEFECTSIZE;
DefectList
1 -50000 -60000 300;
2 70000 80000 400;
3 90000 -10000 500;
EndOfFile;
""")


# ===================================================================
# parse_klarf_string tests
# ===================================================================


class TestParseKlarfString:
    """Parse KLARF content from strings."""

    def test_parse_synthetic_klarf(self):
        """Parse synthetic KLARF → valid list of wafer dicts."""
        wafers = parse_klarf_string(SAMPLE_KLARF)
        assert len(wafers) >= 1
        w = wafers[0]
        assert 'x' in w
        assert 'y' in w
        assert len(w['x']) == 5

    def test_coordinates_extracted(self):
        """x, y coordinates are extracted correctly in mm."""
        wafers = parse_klarf_string(SAMPLE_KLARF)
        w = wafers[0]
        # KLARF coordinates are in µm, should be converted to mm
        # First defect: XREL=50000 µm → 50.0 mm
        assert abs(w['x'][0] - 50.0) < 0.01
        assert abs(w['y'][0] - 30.0) < 0.01

    def test_attributes_populated(self):
        """size and classcode attributes are extracted."""
        wafers = parse_klarf_string(SAMPLE_KLARF)
        w = wafers[0]
        assert 'size' in w
        assert 'classcode' in w
        assert len(w['size']) == 5
        # First defect: DEFECTSIZE=150
        assert w['size'][0] == 150

    def test_wafer_metadata(self):
        """Wafer ID and lot info included."""
        wafers = parse_klarf_string(SAMPLE_KLARF)
        w = wafers[0]
        assert w.get('wafer_id') == 'W01'
        assert w.get('lot_id') == 'LOT001'

    def test_multi_wafer_parsing(self):
        """Multi-wafer KLARF → separate wafer entries."""
        wafers = parse_klarf_string(SAMPLE_KLARF_TWO_WAFERS)
        assert len(wafers) == 2
        assert len(wafers[0]['x']) == 2
        assert len(wafers[1]['x']) == 3

    def test_wafer_diameter_from_sample_size(self):
        """Wafer diameter extracted from SampleSize field."""
        wafers = parse_klarf_string(SAMPLE_KLARF)
        assert wafers[0].get('diameter_mm') == 300.0


# ===================================================================
# load_klarf tests
# ===================================================================


class TestLoadKlarf:
    """Load KLARF files into WaferMap objects."""

    def test_load_klarf_synthetic(self):
        """Parse a synthetic KLARF file → valid WaferMap."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.klarf', delete=False) as f:
            f.write(SAMPLE_KLARF)
            path = f.name
        try:
            wafer = load_klarf(path)
            assert isinstance(wafer, WaferMap)
            assert wafer.n_defects == 5
        finally:
            os.unlink(path)

    def test_klarf_coordinates_in_mm(self):
        """Coordinates are in mm from wafer centre."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.klarf', delete=False) as f:
            f.write(SAMPLE_KLARF)
            path = f.name
        try:
            wafer = load_klarf(path)
            coords = wafer.coordinates
            # First defect at (50, 30) mm
            assert abs(coords[0, 0] - 50.0) < 0.01
            assert abs(coords[0, 1] - 30.0) < 0.01
        finally:
            os.unlink(path)

    def test_klarf_wafer_id_selection(self):
        """Multi-wafer file: select by wafer_id."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.klarf', delete=False) as f:
            f.write(SAMPLE_KLARF_TWO_WAFERS)
            path = f.name
        try:
            w1 = load_klarf(path, wafer_id=0)
            w2 = load_klarf(path, wafer_id=1)
            assert w1.n_defects == 2
            assert w2.n_defects == 3
        finally:
            os.unlink(path)

    def test_klarf_missing_file_raises(self):
        """FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_klarf('/nonexistent/path/file.klarf')

    def test_klarf_malformed_raises(self):
        """Clear error for unparseable content."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.klarf', delete=False) as f:
            f.write('This is not a KLARF file\nRandom garbage\n')
            path = f.name
        try:
            with pytest.raises(ValueError, match='[Kk][Ll][Aa][Rr][Ff]|parse|defect'):
                load_klarf(path)
        finally:
            os.unlink(path)

    def test_klarf_attributes_on_wafer(self):
        """size and classcode are available as WaferMap attributes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.klarf', delete=False) as f:
            f.write(SAMPLE_KLARF)
            path = f.name
        try:
            wafer = load_klarf(path)
            assert 'size' in wafer.attribute_names
            assert 'classcode' in wafer.attribute_names
            sizes = wafer.get_attribute('size')
            assert len(sizes) == 5
        finally:
            os.unlink(path)

    def test_wafer_diameter_set(self):
        """Wafer diameter is set from KLARF SampleSize."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.klarf', delete=False) as f:
            f.write(SAMPLE_KLARF)
            path = f.name
        try:
            wafer = load_klarf(path)
            assert wafer.geometry.diameter_mm == 300.0
        finally:
            os.unlink(path)


# ===================================================================
# WaferMap.from_klarf tests
# ===================================================================


class TestWaferMapFromKlarf:
    """WaferMap class method for KLARF loading."""

    def test_from_klarf_classmethod(self):
        """WaferMap.from_klarf() produces a valid WaferMap."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.klarf', delete=False) as f:
            f.write(SAMPLE_KLARF)
            path = f.name
        try:
            wafer = WaferMap.from_klarf(path)
            assert isinstance(wafer, WaferMap)
            assert wafer.n_defects == 5
        finally:
            os.unlink(path)
