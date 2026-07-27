from pathlib import Path

from file_utils import copy_file_safely, is_generated_output_directory


def test_generated_output_directory_detection_is_case_insensitive():
    assert is_generated_output_directory("transects_output")
    assert is_generated_output_directory("TRANSECTS_OUTPUT_BATCH")
    assert is_generated_output_directory("geotagged_survey")
    assert is_generated_output_directory("Geotagged")
    assert is_generated_output_directory("Extracted_Images")
    assert is_generated_output_directory("Extracted_Images_20260727")
    assert not is_generated_output_directory("survey_images")


def test_copy_retains_basename_when_available(tmp_path):
    source = tmp_path / "source" / "image.jpg"
    source.parent.mkdir()
    source.write_bytes(b"first")
    output = tmp_path / "output"

    result = copy_file_safely(source, output, source_root=source.parent)

    assert result.copied
    assert not result.renamed_for_collision
    assert Path(result.destination_path).name == "image.jpg"
    assert Path(result.destination_path).read_bytes() == b"first"


def test_copy_renames_different_files_with_the_same_basename(tmp_path):
    source_root = tmp_path / "source"
    first = source_root / "camera_a" / "image.jpg"
    second = source_root / "camera_b" / "image.jpg"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    output = tmp_path / "output"

    first_result = copy_file_safely(first, output, source_root=source_root)
    second_result = copy_file_safely(second, output, source_root=source_root)
    repeat_result = copy_file_safely(second, output, source_root=source_root)

    assert Path(first_result.destination_path).name == "image.jpg"
    assert second_result.copied
    assert second_result.renamed_for_collision
    assert Path(second_result.destination_path).name.startswith("image__")
    assert Path(second_result.destination_path).read_bytes() == b"second"
    assert not repeat_result.copied
    assert repeat_result.destination_path == second_result.destination_path
