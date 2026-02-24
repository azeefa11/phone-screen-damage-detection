import streamlit as st
import cv2
import numpy as np

st.title("Damage Detection Platform")

uploaded_file = st.file_uploader("Upload Phone Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # --- Phase 2: Image Handling & Preprocessing ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    output_img = image.copy()
    
    # Preprocessing to reduce wood grain noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # --- Phase 3: Detection Logic (ROI Isolation) ---
    # Find the phone body and ignore the table background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if cnts:
        # Assume the largest rectangular object is the phone
        phone_cnt = max(cnts, key=cv2.contourArea)
        
        # Create a mask so detection stays inside the phone boundaries
        mask = np.zeros(gray.shape, dtype="uint8")
        cv2.drawContours(mask, [phone_cnt], -1, 255, -1)
        
        # Isolate the phone pixels
        phone_only = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Detect sharp edges (cracks/scratches) only within the mask
        edges = cv2.Canny(phone_only, 100, 200) 
        
        damage_cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        damage_count = 0
        
        for c in damage_cnts:
            # Filter out tiny noise and ignore the very outer edge of the phone
            if 20 < cv2.contourArea(c) < 500: 
                # Draw the bounding boxes or contours visually
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                damage_count += 1

        # --- Phase 4: Interface & Output ---
        st.subheader("Analysis Results")
        st.image(output_img, channels="BGR", caption="Detected Damage")

        # Display Summary
        if damage_count > 0:
            st.warning(f"Detected {damage_count} potential damage zones.")
        else:
            st.success("No significant damage detected within the phone area.")
    else:
        st.error("Phone not detected. Please ensure it is on a contrasting surface.")