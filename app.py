import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64
from torchvision import transforms
import torch.nn.functional as F
from functools import lru_cache

# Load the pretrained model
@lru_cache(maxsize=None)
def load_model():
    """Load the pretrained brain segmentation model"""
    try:
        model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch',
            'unet',
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
            force_reload=False
        )
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if model:
    model = model.to(device)

def preprocess_image(image):
    """Preprocess the input image for the model"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to 256x256 (model's expected input size)
    image = image.resize((256, 256), Image.Resampling.LANCZOS)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image

def create_overlay_visualization(original_img, mask, alpha=0.6):
    """Create an overlay visualization of the segmentation"""
    # Convert original image to numpy array
    original_np = np.array(original_img)

    # Create colored mask (red for tumor regions)
    colored_mask = np.zeros_like(original_np)
    colored_mask[:, :, 0] = mask * 255  # Red channel for tumor

    # Create overlay
    overlay = cv2.addWeighted(original_np, 1-alpha, colored_mask, alpha, 0)

    return overlay

def predict_tumor(image):
    """Main prediction function"""
    if model is None:
        return None, "❌ Model failed to load. Please try again."

    if image is None:
        return None, "⚠️ Please upload an image first."

    try:
        # Preprocess the image
        input_tensor, original_img = preprocess_image(image)
        input_tensor = input_tensor.to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            # Apply sigmoid to get probability map
            prediction = torch.sigmoid(prediction)
            # Convert to numpy
            prediction = prediction.squeeze().cpu().numpy()

        # Threshold the prediction (you can adjust this threshold)
        threshold = 0.5
        binary_mask = (prediction > threshold).astype(np.uint8)

        # Create visualizations
        # 1. Original image
        original_array = np.array(original_img)

        # 2. Segmentation mask
        mask_colored = np.zeros((256, 256, 3), dtype=np.uint8)
        mask_colored[:, :, 0] = binary_mask * 255  # Red channel

        # 3. Overlay
        overlay = create_overlay_visualization(original_img, binary_mask, alpha=0.4)

        # 4. Side-by-side comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_array)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(mask_colored)
        axes[1].set_title('Tumor Segmentation', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Red = Tumor)', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        # Save plot to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        # Convert to PIL Image
        result_image = Image.open(buf)

        # Calculate tumor statistics
        total_pixels = 256 * 256
        tumor_pixels = np.sum(binary_mask)
        tumor_percentage = (tumor_pixels / total_pixels) * 100

        # Create analysis report
        analysis_text = f"""
        ## 🧠 Brain Tumor Segmentation Analysis

        **📊 Tumor Statistics:**
        - Total pixels analyzed: {total_pixels:,}
        - Tumor pixels detected: {tumor_pixels:,}
        - Tumor area percentage: {tumor_percentage:.2f}%

        **🎯 Model Performance:**
        - Model: U-Net with attention mechanism
        - Input resolution: 256×256 pixels
        - Detection threshold: {threshold}

        **⚠️ Medical Disclaimer:**
        This is an AI tool for research purposes only.
        Always consult qualified medical professionals for diagnosis.
        """

        return result_image, analysis_text

    except Exception as e:
        error_msg = f"❌ Error during prediction: {str(e)}"
        return None, error_msg

def clear_all():
    """Clear all inputs and outputs"""
    return None, None, ""

# Custom CSS for better styling
css = """
#main-container {
    max-width: 1200px;
    margin: 0 auto;
}
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
#upload-box {
    border: 2px dashed #ccc;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin: 10px 0;
}
.output-image {
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
"""

# Create Gradio interface
with gr.Blocks(css=css, title="Brain Tumor Segmentation") as app:

    # Header
    gr.HTML("""
    <div id="title">
        <h1>🧠 Brain Tumor Segmentation AI</h1>
        <p>Upload an MRI brain scan to detect and visualize tumor regions using deep learning</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h3>📤 Input Image</h3>")

            # Image input with camera option
            image_input = gr.Image(
                label="Upload Brain MRI Scan",
                type="pil",
                sources=["upload", "webcam"],  # Allow both upload and camera
                height=300
            )

            with gr.Row():
                predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")

            gr.HTML("""
            <div style="margin-top: 20px; padding: 15px; background-color: #f0f8ff; border-radius: 8px;">
                <h4>📋 Instructions:</h4>
                <ul>
                    <li>Upload a brain MRI scan image</li>
                    <li>Supported formats: PNG, JPG, JPEG</li>
                    <li>For best results, use clear, high-contrast MRI images</li>
                    <li>You can also use the camera to capture an image from your device</li>
                </ul>
            </div>
            """)

        with gr.Column(scale=2):
            gr.HTML("<h3>📊 Segmentation Results</h3>")

            # Output image
            output_image = gr.Image(
                label="Segmentation Results",
                type="pil",
                height=400,
                elem_classes=["output-image"]
            )

            # Analysis text
            analysis_output = gr.Markdown(
                label="Analysis Report",
                value="Upload an image and click 'Analyze Image' to see results."
            )

    # Add footer with information
    gr.HTML("""
    <div style="margin-top: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 10px;">
        <h4>🔬 About This Tool</h4>
        <p><strong>Model:</strong> Pre-trained U-Net architecture optimized for brain tumor segmentation</p>
        <p><strong>Technology:</strong> PyTorch, Deep Learning, Computer Vision</p>
        <p><strong>Dataset:</strong> Trained on medical MRI brain scans</p>

        <h4>⚠️ Important Medical Disclaimer</h4>
        <p style="color: #d73027; font-weight: bold;">
        This AI tool is for research and educational purposes only. It should NOT be used for medical diagnosis.
        Always consult qualified healthcare professionals for medical advice and diagnosis.
        </p>

        <p style="text-align: center; margin-top: 20px; color: #666;">
        Made with ❤️ using Gradio • Powered by PyTorch • Hosted on 🤗 Hugging Face Spaces
        </p>
    </div>
    """)

    # Event handlers
    predict_btn.click(
        fn=predict_tumor,
        inputs=[image_input],
        outputs=[output_image, analysis_output]
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, output_image, analysis_output]
    )

    # Auto-predict when image is uploaded
    image_input.change(
        fn=predict_tumor,
        inputs=[image_input],
        outputs=[output_image, analysis_output]
    )

# Launch the app
if __name__ == "__main__":
    app.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
