"""Tests for cpp_manager.py — image stacking, utility functions."""

from unittest import mock
import pytest


class TestStackImages:
    """Tests for stack_images()."""

    def test_empty_raises(self):
        from llm_server.inference_server.cpp_manager import stack_images

        with pytest.raises(ValueError, match="不能为空"):
            stack_images([])

    def test_single_image_returns_same(self, sample_image):
        from llm_server.inference_server.cpp_manager import stack_images

        result = stack_images([sample_image])
        assert result is sample_image

    def test_two_images_horizontal(self, sample_image):
        from llm_server.inference_server.cpp_manager import stack_images
        from PIL import Image

        second = Image.new("RGB", (64, 64), color=(0, 255, 0))
        result = stack_images([sample_image, second])

        assert result.size == (128, 64)  # w * 2, h
        # Left half should be red, right half green
        assert result.getpixel((0, 0)) == (255, 0, 0)
        assert result.getpixel((70, 0)) == (0, 255, 0)

    def test_three_images_2x2_black_quadrant(self, sample_image):
        from llm_server.inference_server.cpp_manager import stack_images
        from PIL import Image

        second = Image.new("RGB", (64, 64), color=(0, 255, 0))
        third = Image.new("RGB", (64, 64), color=(0, 0, 255))
        result = stack_images([sample_image, second, third])

        assert result.size == (128, 128)  # w * 2, h * 2
        # Bottom-right should be black (no image placed there)
        assert result.getpixel((70, 70)) == (0, 0, 0)

    def test_four_images_2x2_full(self):
        from llm_server.inference_server.cpp_manager import stack_images
        from PIL import Image

        images = [
            Image.new("RGB", (100, 80), color=(255, 0, 0)),
            Image.new("RGB", (100, 80), color=(0, 255, 0)),
            Image.new("RGB", (100, 80), color=(0, 0, 255)),
            Image.new("RGB", (100, 80), color=(255, 255, 0)),
        ]
        result = stack_images(images)

        assert result.size == (200, 160)
        assert result.getpixel((10, 10)) == (255, 0, 0)
        assert result.getpixel((110, 10)) == (0, 255, 0)
        assert result.getpixel((10, 90)) == (0, 0, 255)
        assert result.getpixel((110, 90)) == (255, 255, 0)

    def test_more_than_four_uses_only_first_four(self):
        from llm_server.inference_server.cpp_manager import stack_images
        from PIL import Image

        images = [Image.new("RGB", (50, 50), color=(c * 40, 0, 0)) for c in range(6)]
        result = stack_images(images)

        assert result.size == (100, 100)
        # Fifth image (index 4) should not be in layout
        # Bottom-right (index 3) should be at (60, 60)
        assert result.getpixel((60, 60)) == (120, 0, 0)


class TestGetLocalIp:
    """Tests for get_local_ip()."""

    def test_returns_ip_string(self, monkeypatch):
        from llm_server.inference_server.cpp_manager import get_local_ip

        mock_sock = mock.MagicMock()
        mock_sock.getsockname.return_value = ("10.0.0.1", 12345)
        monkeypatch.setattr("socket.socket", lambda *a, **kw: mock_sock)

        ip = get_local_ip()
        assert ip == "10.0.0.1"
