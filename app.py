import streamlit as st
import cv2
import numpy as np

st.title("Phone Screen Damage Detection (Improved Version)")

uploaded_file = st.file_uploader("Upload Phone Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    original = image.copy()

    st.subheader("Original Image")
    st.image(original, channels="BGR")

    # -----------------------
    # Preprocessing
    # -----------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement (important for scratches)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # -----------------------
    # Contour Detection
    # -----------------------

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    damage_count = 0
    image_area = image.shape[0] * image.shape[1]

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore very small noise
        if area < 80:
            continue

        # Ignore very large regions (like phone border)
        if area > image_area * 0.2:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / float(h)

        # Detect long thin regions (likely scratches)
        if aspect_ratio > 3 or aspect_ratio < 0.3:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            damage_count += 1

    # -----------------------
    # Output
    # -----------------------

    st.subheader("Detected Damage")
    st.image(image, channels="BGR")

    st.subheader("Damage Summary")

    if damage_count == 0:
        st.success("No visible damage detected.")
    elif damage_count <= 3:
        st.warning("Minor scratch-like damage detected.")
    else:
        st.error("Multiple scratch/crack patterns detected. Possible moderate damage.")

    st.write("Total detected damage regions:", damage_count)