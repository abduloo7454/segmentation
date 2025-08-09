import streamlit as st
import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.title("Medical Image Segmentation with Pre-trained FCN ResNet50")

st.write("""
Upload a medical image (jpg, jpeg, png, bmp) and the app will perform segmentation using a pre-trained FCN ResNet50 model from PyTorch/Torchvision.
Note: This model is trained on COCO/VOC dataset (general objects), so segmentation results on medical images may not be medically accurate but will show model operation.
""")

uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "bmp"])

@st.cache_resource
def load_model_and_weights():
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model.eval()
    return model, weights

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model, weights = load_model_and_weights()
    preprocess = weights.transforms()
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)["out"][0]
    pred_mask = output.argmax(0).byte().cpu().numpy()

    # Define a color map for visualization (up to 20 classes)
    num_classes = len(weights.meta['categories'])
    base_colors = plt.get_cmap("tab20").colors  # 20 colors, repeated if more classes
    colormap = (np.array(base_colors) * 255).astype(np.uint8)

    seg_image = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for label_index in range(num_classes):
        color = colormap[label_index % len(colormap)]
        seg_image[pred_mask == label_index] = color

    seg_image_pil = Image.fromarray(seg_image)
    st.image(seg_image_pil, caption="Segmentation Mask", use_column_width=True)

    # Optional: show category legend
    legend_items = []
    categories = weights.meta['categories']
    for i in range(num_classes):
        color_hex = '#%02x%02x%02x' % tuple(colormap[i % len(colormap)])
        legend_items.append(f'<span style="background-color:{color_hex};padding:2px 8px;border-radius:3px;">{categories[i]}</span>')
    st.markdown("**Category Color Legend:**<br>" + " ".join(legend_items), unsafe_allow_html=True)
else:
    st.info("Please upload an image to segment.")
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
