"""Tests for batch inference functionality."""

import tempfile
from pathlib import Path

import pytest

from groundeddino_vl.batch_inference import (
    DEFAULT_WILDLIFE_CLASSES,
    export_coco_json,
    export_csv,
    export_labelstudio_json,
    export_yolo_format,
    find_images,
)


def test_default_wildlife_classes():
    """Test that default wildlife classes are properly defined."""
    assert len(DEFAULT_WILDLIFE_CLASSES) > 0
    assert "bear" in DEFAULT_WILDLIFE_CLASSES
    assert "wolf" in DEFAULT_WILDLIFE_CLASSES
    assert "cougar" in DEFAULT_WILDLIFE_CLASSES
    assert isinstance(DEFAULT_WILDLIFE_CLASSES, list)


def test_find_images():
    """Test image file discovery."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test images
        (tmp_path / "test1.jpg").touch()
        (tmp_path / "test2.png").touch()
        (tmp_path / "test3.JPG").touch()  # Test case insensitivity
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "test4.jpg").touch()
        (tmp_path / "notimage.txt").touch()  # Should be ignored

        images = find_images(tmp_path)

        assert len(images) == 4
        assert all(img.suffix.lower() in [".jpg", ".jpeg", ".png"] for img in images)


def test_export_coco_json():
    """Test COCO JSON export."""
    results = [
        {
            "filename": "test1.jpg",
            "filepath": "/path/test1.jpg",
            "width": 1920,
            "height": 1080,
            "detections": [
                {"xyxy": [100, 200, 300, 400], "score": 0.95, "label": "bear"},
                {"xyxy": [500, 600, 700, 800], "score": 0.85, "label": "wolf"},
            ],
        }
    ]
    classes = ["bear", "wolf", "cougar"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = Path(f.name)

    try:
        export_coco_json(results, output_path, classes)
        assert output_path.exists()

        import json

        with open(output_path) as f:
            coco = json.load(f)

        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 2
        assert len(coco["categories"]) == 3

        # Check category mapping
        category_names = {cat["id"]: cat["name"] for cat in coco["categories"]}
        assert category_names[1] == "bear"
        assert category_names[2] == "wolf"

    finally:
        output_path.unlink()


def test_export_yolo_format():
    """Test YOLO format export."""
    results = [
        {
            "filename": "test1.jpg",
            "filepath": "/path/test1.jpg",
            "width": 1000,
            "height": 1000,
            "detections": [
                {"xyxy": [250, 250, 750, 750], "score": 0.95, "label": "bear"},
            ],
        }
    ]
    classes = ["bear", "wolf"]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        export_yolo_format(results, output_dir, classes)

        # Check classes.txt
        classes_file = output_dir / "classes.txt"
        assert classes_file.exists()
        with open(classes_file) as f:
            class_list = f.read().strip().split("\n")
        assert class_list == ["bear", "wolf"]

        # Check label file
        label_file = output_dir / "labels" / "test1.txt"
        assert label_file.exists()

        with open(label_file) as f:
            line = f.read().strip()

        # Parse YOLO format: class_id x_center y_center width height
        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # bear is class 0
        # Center should be at 0.5, 0.5 (middle of 1000x1000 image)
        assert abs(float(parts[1]) - 0.5) < 0.01
        assert abs(float(parts[2]) - 0.5) < 0.01
        # Size should be 0.5, 0.5 (half of image)
        assert abs(float(parts[3]) - 0.5) < 0.01
        assert abs(float(parts[4]) - 0.5) < 0.01


def test_export_labelstudio_json():
    """Test Label Studio JSON export."""
    results = [
        {
            "filename": "test1.jpg",
            "filepath": "/path/test1.jpg",
            "width": 1000,
            "height": 1000,
            "detections": [
                {"xyxy": [100, 200, 300, 400], "score": 0.95, "label": "bear"},
            ],
            "avg_score": 0.95,
            "model_version": "1.0.0",
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_path = Path(f.name)

    try:
        export_labelstudio_json(results, output_path)
        assert output_path.exists()

        import json

        with open(output_path) as f:
            ls_tasks = json.load(f)

        assert len(ls_tasks) == 1
        task = ls_tasks[0]
        assert "data" in task
        assert "predictions" in task
        assert len(task["predictions"]) == 1

        prediction = task["predictions"][0]
        assert "result" in prediction
        assert len(prediction["result"]) == 1

        annotation = prediction["result"][0]
        assert annotation["value"]["rectanglelabels"] == ["bear"]
        assert annotation["score"] == 0.95

    finally:
        output_path.unlink()


def test_export_csv():
    """Test CSV export."""
    results = [
        {
            "filename": "test1.jpg",
            "width": 1920,
            "height": 1080,
            "detections": [
                {"xyxy": [100.5, 200.25, 300.75, 400.0], "score": 0.95, "label": "bear"},
                {"xyxy": [500, 600, 700, 800], "score": 0.85, "label": "wolf"},
            ],
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        output_path = Path(f.name)

    try:
        export_csv(results, output_path)
        assert output_path.exists()

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 3  # header + 2 detections
        assert "filename,label,score,x1,y1,x2,y2,width,height" in lines[0]
        assert "test1.jpg,bear,0.9500" in lines[1]
        assert "test1.jpg,wolf,0.8500" in lines[2]

    finally:
        output_path.unlink()


@pytest.mark.parametrize(
    "classes_input,expected",
    [
        ("bear,wolf,cougar", ["bear", "wolf", "cougar"]),
        ("bear, wolf , cougar", ["bear", "wolf", "cougar"]),  # Test whitespace handling
        ("bear", ["bear"]),
        (None, DEFAULT_WILDLIFE_CLASSES),
    ],
)
def test_class_parsing(classes_input, expected):
    """Test class list parsing from command line."""
    if classes_input:
        parsed = [c.strip() for c in classes_input.split(",") if c.strip()]
        assert parsed == expected
    else:
        assert expected == DEFAULT_WILDLIFE_CLASSES


def test_batch_inference_imports():
    """Test that batch inference module can be imported."""
    try:
        from groundeddino_vl import batch_inference

        assert hasattr(batch_inference, "main")
        assert hasattr(batch_inference, "process_batch")
        assert hasattr(batch_inference, "find_images")
    except ImportError as e:
        pytest.fail(f"Failed to import batch_inference: {e}")


def test_supported_image_formats():
    """Test that common image formats are supported."""
    from groundeddino_vl.batch_inference import SUPPORTED_IMAGE_FORMATS

    assert ".jpg" in SUPPORTED_IMAGE_FORMATS
    assert ".jpeg" in SUPPORTED_IMAGE_FORMATS
    assert ".png" in SUPPORTED_IMAGE_FORMATS
    assert ".bmp" in SUPPORTED_IMAGE_FORMATS


def test_find_images_empty_directory():
    """Test find_images with empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        images = find_images(Path(tmpdir))
        assert len(images) == 0


def test_find_images_recursive():
    """Test that find_images searches recursively."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create nested structure
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "level1" / "level2" / "deep.jpg").touch()

        images = find_images(tmp_path)
        assert len(images) == 1
        assert images[0].name == "deep.jpg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
