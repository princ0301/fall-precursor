from pathlib import Path

from fallprecursor.data.le2i import parse_le2i_annotation

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_parses_fall_onset_and_impact_as_zero_indexed():
    onset_frame, impact_frame = parse_le2i_annotation(FIXTURES_DIR / "le2i_fall_example.txt")
    assert onset_frame == 210
    assert impact_frame == 237


def test_zero_zero_header_means_no_fall():
    onset_frame, impact_frame = parse_le2i_annotation(FIXTURES_DIR / "le2i_no_fall_example.txt")
    assert onset_frame is None
    assert impact_frame is None


def test_missing_header_means_no_fall():
    onset_frame, impact_frame = parse_le2i_annotation(FIXTURES_DIR / "le2i_no_header_example.txt")
    assert onset_frame is None
    assert impact_frame is None