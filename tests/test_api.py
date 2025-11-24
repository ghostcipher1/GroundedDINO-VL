"""
Unit tests for the GroundedDINO-VL public API

Tests the core functionality of the simplified public API including:
- Model loading
- Image preprocessing
- Inference
- Result handling

Author: GroundedDINO-VL Team
License: Apache 2.0
"""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from groundeddino_vl import (
    DetectionResult,
    annotate,
    load_model,
    predict,
    preprocess_image,
)


class TestDetectionResult(unittest.TestCase):
    """Tests for the DetectionResult dataclass"""

    def setUp(self):
        """Set up test fixtures"""
        self.boxes = torch.tensor(
            [
                [0.5, 0.5, 0.2, 0.3],  # cxcywh format
                [0.3, 0.7, 0.1, 0.2],
            ]
        )
        self.labels = ["car", "person"]
        self.scores = torch.tensor([0.95, 0.87])
        self.image_size = (480, 640)  # H, W

    def test_creation(self):
        """Test creating a DetectionResult"""
        result = DetectionResult(
            boxes=self.boxes, labels=self.labels, scores=self.scores, image_size=self.image_size
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result.labels, ["car", "person"])
        torch.testing.assert_close(result.boxes, self.boxes)
        torch.testing.assert_close(result.scores, self.scores)

    def test_to_xyxy_normalized(self):
        """Test converting boxes to xyxy format (normalized)"""
        result = DetectionResult(
            boxes=self.boxes, labels=self.labels, scores=self.scores, image_size=self.image_size
        )

        boxes_xyxy = result.to_xyxy(denormalize=False)

        # Should be in xyxy format but still normalized [0, 1]
        self.assertEqual(boxes_xyxy.shape, (2, 4))
        # First box: cx=0.5, cy=0.5, w=0.2, h=0.3
        # -> x1=0.4, y1=0.35, x2=0.6, y2=0.65
        expected_first = torch.tensor([0.4, 0.35, 0.6, 0.65])
        torch.testing.assert_close(boxes_xyxy[0], expected_first, rtol=1e-5, atol=1e-5)

    def test_to_xyxy_denormalized(self):
        """Test converting boxes to pixel coordinates"""
        result = DetectionResult(
            boxes=self.boxes, labels=self.labels, scores=self.scores, image_size=self.image_size
        )

        boxes_xyxy = result.to_xyxy(denormalize=True)

        # Should be in pixel coordinates
        # First box normalized: [0.4, 0.35, 0.6, 0.65]
        # Image size: 480h x 640w
        # Expected: [0.4*640, 0.35*480, 0.6*640, 0.65*480]
        #         = [256, 168, 384, 312]
        expected_first = torch.tensor([256.0, 168.0, 384.0, 312.0])
        torch.testing.assert_close(boxes_xyxy[0], expected_first, rtol=1e-4, atol=1e-4)

    def test_len(self):
        """Test __len__ method"""
        result = DetectionResult(boxes=self.boxes, labels=self.labels, scores=self.scores)
        self.assertEqual(len(result), 2)

    def test_repr(self):
        """Test __repr__ method"""
        result = DetectionResult(boxes=self.boxes, labels=self.labels, scores=self.scores)
        repr_str = repr(result)
        self.assertIn("DetectionResult", repr_str)
        self.assertIn("car", repr_str)


class TestPreprocessImage(unittest.TestCase):
    """Tests for image preprocessing functions"""

    def test_preprocess_numpy_array(self):
        """Test preprocessing a numpy array"""
        # Create a dummy RGB image
        image_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        tensor = preprocess_image(image_np)

        # Should be a tensor with shape (3, H, W)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape[0], 3)  # 3 channels
        # Height/width may be resized
        self.assertLessEqual(max(tensor.shape[1:]), 1333)

    def test_preprocess_pil_image(self):
        """Test preprocessing a PIL image"""
        image_pil = Image.new("RGB", (640, 480), color="red")

        tensor = preprocess_image(image_pil)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape[0], 3)

    def test_preprocess_tensor_passthrough(self):
        """Test that tensors are passed through"""
        tensor_in = torch.randn(3, 480, 640)

        tensor_out = preprocess_image(tensor_in)

        # Should be the same object
        self.assertIs(tensor_in, tensor_out)

    def test_preprocess_custom_size(self):
        """Test preprocessing with custom size parameters"""
        image_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        tensor = preprocess_image(image_np, max_size=1000, size=600)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertLessEqual(max(tensor.shape[1:]), 1000)


class TestAPIIntegration(unittest.TestCase):
    """Integration tests for the public API (with mocking)"""

    @patch("groundeddino_vl.api.build_model")
    @patch("groundeddino_vl.api.SLConfig")
    @patch("groundeddino_vl.api.torch.load")
    def test_load_model(self, mock_torch_load, mock_slconfig, mock_build_model):
        """Test load_model function"""
        # Set up mocks
        mock_config = MagicMock()
        mock_slconfig.fromfile.return_value = mock_config

        mock_model = MagicMock()
        mock_build_model.return_value = mock_model

        mock_checkpoint = {"model": {"layer1.weight": torch.randn(10, 10)}}
        mock_torch_load.return_value = mock_checkpoint

        # Call load_model
        with (
            tempfile.NamedTemporaryFile(suffix=".py") as config_file,
            tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint_file,
        ):

            load_model(  # noqa: F841
                config_path=config_file.name, checkpoint_path=checkpoint_file.name, device="cpu"
            )

        # Verify behavior
        mock_slconfig.fromfile.assert_called_once()
        mock_build_model.assert_called_once()
        mock_model.load_state_dict.assert_called_once()
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with("cpu")

    def test_predict_with_mock_model(self):
        """Test predict function with a mock model"""
        # Create mock model
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()

        with patch("groundeddino_vl.api._predict_internal") as mock_predict:
            # Mock internal predict to return simple results
            mock_predict.return_value = (
                torch.tensor([[0.5, 0.5, 0.2, 0.3]]),  # boxes
                torch.tensor([0.95]),  # scores
                ["car"],  # labels
            )

            # Create a test image
            image_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Run predict
            result = predict(
                model=mock_model,
                image=image_np,
                text_prompt="car",
                box_threshold=0.3,
                text_threshold=0.25,
                device="cpu",
            )

            # Verify result
            self.assertIsInstance(result, DetectionResult)
            self.assertEqual(len(result), 1)
            self.assertEqual(result.labels[0], "car")
            self.assertEqual(result.image_size, (480, 640))

    def test_predict_with_image_path(self):
        """Test predict with image file path"""
        mock_model = MagicMock()

        with (
            patch("groundeddino_vl.api.load_image") as mock_load_image,
            patch("groundeddino_vl.api._predict_internal") as mock_predict,
        ):

            # Mock load_image
            mock_load_image.return_value = (
                np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                torch.randn(3, 480, 640),
            )

            # Mock predict
            mock_predict.return_value = (
                torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
                torch.tensor([0.95]),
                ["car"],
            )

            # Create temporary image file
            with tempfile.NamedTemporaryFile(suffix=".jpg") as img_file:
                # Create a simple image
                img = Image.new("RGB", (640, 480), color="red")
                img.save(img_file.name)

                result = predict(
                    model=mock_model, image=img_file.name, text_prompt="car", device="cpu"
                )

            self.assertIsInstance(result, DetectionResult)
            mock_load_image.assert_called_once()

    def test_predict_invalid_image_type(self):
        """Test that predict raises TypeError for invalid image types"""
        mock_model = MagicMock()

        with self.assertRaises(TypeError):
            predict(model=mock_model, image=12345, text_prompt="car")  # Invalid type


class TestAnnotate(unittest.TestCase):
    """Tests for the annotate function"""

    @patch("groundeddino_vl.api._annotate_internal")
    def test_annotate(self, mock_annotate_internal):
        """Test annotate function"""
        # Create test data
        image_np = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = DetectionResult(
            boxes=torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
            labels=["car"],
            scores=torch.tensor([0.95]),
            image_size=(480, 640),
        )

        # Mock the internal annotate function
        mock_annotate_internal.return_value = image_np

        # Call annotate
        annotate(image_np, result)

        # Verify it called the internal function correctly
        mock_annotate_internal.assert_called_once_with(
            image_source=image_np, boxes=result.boxes, logits=result.scores, phrases=result.labels
        )


def run_tests():
    """Run all tests"""
    unittest.main(argv=[""], verbosity=2, exit=False)


if __name__ == "__main__":
    unittest.main()
