import streamlit as st
import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np

st.title("Medical Image Segmentation with Pre-trained FCN ResNet50")

st.write("""
Upload a medical image (jpg, jpeg, png, bmp) and the app will perform segmentation using a pre-trained FCN ResNet50 model.
This model is trained on general objects, so segmentation results may vary for medical images. 
""")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load the pre-trained model and weights
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()

    # Preprocess image
    input_tensor = weights.transforms()(image).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)["out"][0]

    # Get predicted mask (argmax over channels)
    pred_mask = output.argmax(0).byte().cpu().numpy()

    # Convert mask to an RGB image for better visualization
    # Map each class label to a color
    palette = weights.meta["categories"]  # class names
    # Using colormap from weights visualization for segmentation masks
    colormap = weights.meta["color_map"]

    seg_image = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for label_index, color in enumerate(colormap):
        seg_image[pred_mask == label_index] = color

    seg_image_pil = Image.fromarray(seg_image)

    st.image(seg_image_pil, caption="Segmentation Mask", use_column_width=True)
else:
    st.info("Please upload an image to segment.")
