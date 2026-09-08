from pathlib import Path


def parse_le2i_annotation(annotation_path: Path) -> tuple[int | None, int | None]:
    """Read a Le2i annotation .txt file and return (onset_frame, impact_frame).

    Fall clips start with two header lines giving the fall start and end
    frame, 1-indexed. No-fall clips are inconsistent in this dataset: some
    use a "0" / "0" header, others omit the header entirely and start
    directly with bounding-box rows. Both cases are treated as no fall.
    Returned frame indices are 0-indexed to match array indexing used
    elsewhere in this project.
    """
    with open(annotation_path) as annotation_file:
        first_line = annotation_file.readline().strip()
        second_line = annotation_file.readline().strip()

    if not (_is_header_line(first_line) and _is_header_line(second_line)):
        return None, None

    onset_frame_1indexed = int(first_line)
    impact_frame_1indexed = int(second_line)

    if onset_frame_1indexed == 0 and impact_frame_1indexed == 0:
        return None, None

    return onset_frame_1indexed - 1, impact_frame_1indexed - 1


def _is_header_line(line: str) -> bool:
    """A header line is a bare frame number, as opposed to a comma-separated bounding-box row."""
    return "," not in line and line.isdigit()