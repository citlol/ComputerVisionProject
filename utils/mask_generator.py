"""
Mask generator for irregular holes
Creates random masks for training and testing
"""
import numpy as np
import cv2
import random


class IrregularMaskGenerator:
    """Generate irregular masks for image inpainting"""

    def __init__(self, height=256, width=256, max_vertex=12, max_angle=4.0,
                 max_length=100, max_brush_width=24, min_area_ratio=0.1, max_area_ratio=0.4):
        """
        Args:
            height, width: Mask dimensions
            max_vertex: Maximum number of vertices for random walks
            max_angle: Maximum turning angle between lines
            max_length: Maximum length of each line
            max_brush_width: Maximum width of brush stroke
            min_area_ratio: Minimum ratio of masked area
            max_area_ratio: Maximum ratio of masked area
        """
        self.height = height
        self.width = width
        self.max_vertex = max_vertex
        self.max_angle = max_angle
        self.max_length = max_length
        self.max_brush_width = max_brush_width
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    def generate_random_walk_mask(self):
        """Generate mask using random walk algorithm"""
        mask = np.zeros((self.height, self.width), dtype=np.float32)

        # Number of strokes
        num_strokes = random.randint(1, 5)

        for _ in range(num_strokes):
            # Random starting point
            start_x = random.randint(0, self.width - 1)
            start_y = random.randint(0, self.height - 1)

            # Number of vertices for this stroke
            num_vertex = random.randint(4, self.max_vertex)

            # Random walk
            for i in range(num_vertex):
                if i == 0:
                    angle = random.random() * 2 * np.pi
                else:
                    angle = angle + (random.random() - 0.5) * self.max_angle

                length = random.randint(10, self.max_length)
                brush_width = random.randint(10, self.max_brush_width)

                end_x = int(start_x + length * np.cos(angle))
                end_y = int(start_y + length * np.sin(angle))

                # Clip to image bounds
                end_x = np.clip(end_x, 0, self.width - 1)
                end_y = np.clip(end_y, 0, self.height - 1)

                # Draw line
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_width)

                # Update starting point
                start_x, start_y = end_x, end_y

        return mask

    def generate_rectangular_mask(self, num_holes=5):
        """Generate mask with random rectangular holes"""
        mask = np.zeros((self.height, self.width), dtype=np.float32)

        for _ in range(num_holes):
            # Random rectangle dimensions
            hole_width = random.randint(20, self.width // 3)
            hole_height = random.randint(20, self.height // 3)

            # Random position
            x = random.randint(0, self.width - hole_width)
            y = random.randint(0, self.height - hole_height)

            mask[y:y+hole_height, x:x+hole_width] = 1.0

        return mask

    def generate_mixed_mask(self):
        """Generate mask combining random walks and rectangles"""
        mask1 = self.generate_random_walk_mask()
        mask2 = self.generate_rectangular_mask(num_holes=random.randint(1, 3))

        # Combine masks
        mask = np.maximum(mask1, mask2)

        # Ensure mask area is within specified ratio
        area_ratio = mask.sum() / (self.height * self.width)
        if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
            # Regenerate
            return self.generate_mixed_mask()

        return mask

    def __call__(self, mode='mixed'):
        """
        Generate a mask
        Args:
            mode: 'random_walk', 'rectangular', or 'mixed'
        Returns:
            mask: Binary mask (H, W) where 1 = hole, 0 = valid
        """
        if mode == 'random_walk':
            mask = self.generate_random_walk_mask()
        elif mode == 'rectangular':
            mask = self.generate_rectangular_mask()
        else:  # mixed
            mask = self.generate_mixed_mask()

        return mask


class CenterMaskGenerator:
    """Generate center square mask for testing"""

    def __init__(self, height=256, width=256, mask_size=128):
        self.height = height
        self.width = width
        self.mask_size = mask_size

    def __call__(self):
        """Generate center square mask"""
        mask = np.zeros((self.height, self.width), dtype=np.float32)

        # Center position
        start_y = (self.height - self.mask_size) // 2
        start_x = (self.width - self.mask_size) // 2

        mask[start_y:start_y+self.mask_size, start_x:start_x+self.mask_size] = 1.0

        return mask
