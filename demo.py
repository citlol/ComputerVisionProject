"""
Demo script for image inpainting
Provides a Gradio web interface and command-line interface
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

from models.partial_conv import PartialConvUNet
from utils.mask_generator import IrregularMaskGenerator, CenterMaskGenerator


class InpaintingDemo:
    def __init__(self, checkpoint_path, device='cuda', image_size=256):
        """
        Initialize the inpainting demo

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu')
            image_size: Image size for processing
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = PartialConvUNet(in_channels=3, out_channels=3, base_filters=64)
        self.model = self.model.to(self.device)

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully!")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
            print("Using randomly initialized model")

        self.model.eval()

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.to_pil = transforms.ToPILImage()

        # Mask generators
        self.irregular_mask_gen = IrregularMaskGenerator(
            height=image_size,
            width=image_size,
            max_vertex=12,
            max_angle=4.0,
            max_length=100,
            max_brush_width=24,
            min_area_ratio=0.1,
            max_area_ratio=0.4
        )

        self.center_mask_gen = CenterMaskGenerator(
            height=image_size,
            width=image_size,
            mask_size=128
        )

    def process_image(self, image, mask=None, mask_type='center'):
        """
        Process a single image

        Args:
            image: PIL Image or numpy array
            mask: Optional custom mask (PIL Image or numpy array)
            mask_type: Type of automatic mask ('center', 'irregular')

        Returns:
            output_image: Inpainted PIL Image
            masked_input: Masked input image for visualization
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype('uint8'), 'RGB')

        # Transform image
        original_size = image.size
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate or process mask
        if mask is not None:
            # Use provided mask
            if isinstance(mask, np.ndarray):
                mask_img = Image.fromarray(mask.astype('uint8'), 'L')
            else:
                mask_img = mask

            mask_tensor = transforms.Resize((self.image_size, self.image_size))(mask_img)
            mask_tensor = transforms.ToTensor()(mask_tensor)
            mask_np = mask_tensor.squeeze().numpy()
        else:
            # Generate automatic mask
            if mask_type == 'center':
                mask_np = self.center_mask_gen()
            else:  # irregular
                mask_np = self.irregular_mask_gen(mode='mixed')

        # Convert mask to tensor (1 = hole, 0 = valid) -> invert for model
        mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        mask_inv = 1 - mask_tensor  # Model expects 1=valid, 0=hole

        # Create masked image for visualization
        masked_img_tensor = img_tensor * mask_inv + mask_tensor

        # Inpaint
        with torch.no_grad():
            output_tensor = self.model(masked_img_tensor, mask_inv)

        # Convert to PIL
        output_image = self.to_pil(output_tensor.squeeze().cpu())
        masked_input = self.to_pil(masked_img_tensor.squeeze().cpu())

        # Resize back to original size
        output_image = output_image.resize(original_size, Image.LANCZOS)
        masked_input = masked_input.resize(original_size, Image.LANCZOS)

        return output_image, masked_input

    def inpaint_with_custom_mask(self, image, mask):
        """Inpaint with custom user-drawn mask"""
        output, masked_input = self.process_image(image, mask=mask)
        return output, masked_input

    def inpaint_auto(self, image, mask_type):
        """Inpaint with automatic mask"""
        output, masked_input = self.process_image(image, mask_type=mask_type.lower())
        return output, masked_input


def create_gradio_interface(demo):
    """Create Gradio web interface"""

    def inpaint_auto_fn(image, mask_type):
        if image is None:
            return None, None
        output, masked = demo.inpaint_auto(image, mask_type)
        return masked, output

    # Create interface with tabs
    with gr.Blocks(title="Image Inpainting Demo") as interface:
        gr.Markdown("# Image Inpainting Demo")
        gr.Markdown("Upload an image and let the model fill in missing regions!")

        with gr.Tab("Automatic Mask"):
            gr.Markdown("### Upload an image and select mask type")

            with gr.Row():
                with gr.Column():
                    input_image_auto = gr.Image(type="pil", label="Input Image")
                    mask_type = gr.Radio(
                        choices=["Center", "Irregular"],
                        value="Center",
                        label="Mask Type"
                    )
                    auto_button = gr.Button("Inpaint", variant="primary")

                with gr.Column():
                    masked_output_auto = gr.Image(label="Masked Input")
                    result_auto = gr.Image(label="Inpainted Result")

            auto_button.click(
                fn=inpaint_auto_fn,
                inputs=[input_image_auto, mask_type],
                outputs=[masked_output_auto, result_auto]
            )

            gr.Examples(
                examples=[
                    ["demo_images/sample1.jpg", "Center"],
                    ["demo_images/sample2.jpg", "Irregular"],
                ],
                inputs=[input_image_auto, mask_type],
                outputs=[masked_output_auto, result_auto],
                fn=inpaint_auto_fn,
                cache_examples=False,
            )

        with gr.Tab("Random Mask"):
            gr.Markdown("### Generate random irregular masks for testing")

            with gr.Row():
                with gr.Column():
                    input_image_random = gr.Image(type="pil", label="Input Image")
                    random_button = gr.Button("Inpaint with Random Mask", variant="primary")

                with gr.Column():
                    masked_output_random = gr.Image(label="Masked Input")
                    result_random = gr.Image(label="Inpainted Result")

            def inpaint_random_fn(image):
                if image is None:
                    return None, None
                output, masked = demo.inpaint_auto(image, "Irregular")
                return masked, output

            random_button.click(
                fn=inpaint_random_fn,
                inputs=[input_image_random],
                outputs=[masked_output_random, result_random]
            )

    return interface


def command_line_inference(args):
    """Command-line inference"""
    demo = InpaintingDemo(args.checkpoint, device=args.device, image_size=args.image_size)

    if args.input_image:
        # Process single image
        image = Image.open(args.input_image).convert('RGB')
        output, masked = demo.process_image(image, mask_type=args.mask_type)

        # Save results
        output_path = args.output or 'inpainted_output.png'
        output.save(output_path)
        print(f"Result saved to {output_path}")

        if args.save_masked:
            masked_path = output_path.replace('.png', '_masked.png')
            masked.save(masked_path)
            print(f"Masked input saved to {masked_path}")
    else:
        print("No input image specified. Use --input_image to process an image.")


def main():
    parser = argparse.ArgumentParser(description='Image Inpainting Demo')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for processing')
    parser.add_argument('--mode', type=str, default='web', choices=['web', 'cli'],
                        help='Mode: web interface or command-line')
    parser.add_argument('--input_image', type=str, default=None,
                        help='Input image path (for CLI mode)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (for CLI mode)')
    parser.add_argument('--mask_type', type=str, default='center',
                        choices=['center', 'irregular'],
                        help='Mask type for automatic masking (for CLI mode)')
    parser.add_argument('--save_masked', action='store_true',
                        help='Save masked input image (for CLI mode)')
    parser.add_argument('--share', action='store_true',
                        help='Create public share link (for web mode)')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port for web interface')

    args = parser.parse_args()

    if args.mode == 'web':
        # Launch web interface
        demo = InpaintingDemo(args.checkpoint, device=args.device, image_size=args.image_size)
        interface = create_gradio_interface(demo)
        interface.launch(share=args.share, server_port=args.port)
    else:
        # Command-line mode
        command_line_inference(args)


if __name__ == '__main__':
    main()
