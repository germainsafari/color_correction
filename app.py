import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import os
import glob

# ==========================================
# 1. COLOR ENGINE (Statistical / Reinhard)
# ==========================================
class ColorMatcher:
    def get_image_stats(self, image):
        # Convert to LAB space (L=Lightness, A=Green/Red, B=Blue/Yellow)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
        (l, a, b) = cv2.split(image_lab)
        return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())

    def find_best_reference(self, source_img, reference_images):
        src_stats = self.get_image_stats(source_img)
        src_l_mean = src_stats[0]
        src_a_mean = src_stats[2]
        src_b_mean = src_stats[4]
        
        best_ref = None
        min_diff = float('inf')
        best_ref_name = ""
        
        for name, ref_img in reference_images.items():
            ref_stats = self.get_image_stats(ref_img)
            ref_l_mean = ref_stats[0]
            ref_a_mean = ref_stats[2]
            ref_b_mean = ref_stats[4]
            
            # Weighted Euclidean Distance
            # Higher weight on Luminance (2.0) to avoid exposure mismatches
            diff_l = (src_l_mean - ref_l_mean) ** 2
            diff_a = (src_a_mean - ref_a_mean) ** 2
            diff_b = (src_b_mean - ref_b_mean) ** 2
            
            total_diff = np.sqrt((diff_l * 2.0) + diff_a + diff_b)
            
            if total_diff < min_diff:
                min_diff = total_diff
                best_ref = ref_img
                best_ref_name = name
                
        return best_ref, best_ref_name

    def apply_smart_transfer(self, source, target, use_auto_contrast=True):
        """
        Applies color correction. 
        If use_auto_contrast is True, it performs dynamic range stretching.
        """
        # Convert to LAB space
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        (l_src, a_src, b_src) = cv2.split(source_lab)
        (l_tar, a_tar, b_tar) = cv2.split(target_lab)

        # Calculate Statistics
        l_mean_src, l_std_src = l_src.mean(), l_src.std()
        a_mean_src, a_std_src = a_src.mean(), a_src.std()
        b_mean_src, b_std_src = b_src.mean(), b_src.std()

        l_mean_tar, l_std_tar = l_tar.mean(), l_tar.std()
        a_mean_tar, a_std_tar = a_tar.mean(), a_tar.std()
        b_mean_tar, b_std_tar = b_tar.mean(), b_tar.std()

        eps = 1e-5
        
        # 1. Color (Chroma - A/B Channels): Aggressive match
        a_new = ((a_src - a_mean_src) * (a_std_tar / (a_std_src + eps))) + a_mean_tar
        b_new = ((b_src - b_mean_src) * (b_std_tar / (b_std_src + eps))) + b_mean_tar

        # 2. Lightness (Luma - L Channel): Soft Transfer
        # 80% Original Contrast / 20% Reference Contrast
        contrast_blend = (l_std_src * 0.80) + (l_std_tar * 0.20)
        
        l_new = ((l_src - l_mean_src) * (contrast_blend / (l_std_src + eps))) + l_mean_tar

        # --- STEP 3: DYNAMIC RANGE RECOVERY (OPTIONAL) ---
        l_final = l_new
        
        if use_auto_contrast:
            # Get the darkest (1%) and brightest (99%) pixel values
            min_val = np.percentile(l_new, 1)
            max_val = np.percentile(l_new, 99)
            
            # Min-Max Normalization (Stretching logic)
            scale = 255.0 / (max_val - min_val + eps)
            l_stretched = (l_new - min_val) * scale
            
            # Final Blend: 30% Stretched result with 70% Reinhard result.
            l_final = (l_stretched * 0.3) + (l_new * 0.7)
        # ----------------------------------------------------

        # Final Clipping
        l_final = np.clip(l_final, 0, 255)
        a_new = np.clip(a_new, 0, 255)
        b_new = np.clip(b_new, 0, 255)

        # Merge and convert back
        transfer_lab = cv2.merge([l_final, a_new, b_new])
        transfer_bgr = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
        
        return transfer_bgr

# ==========================================
# 2. AI ENGINE (Face Detection + Segmentation)
# ==========================================
class HumanDetector:
    def __init__(self):
        # 1. Face Detector (The Gatekeeper)
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)
        
        # 2. Body Segmenter (The Masker)
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

    def has_face(self, image_rgb):
        """Checks if there is at least one visible face in the image."""
        results = self.face_detector.process(image_rgb)
        return results.detections is not None

    def get_mask(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Gatekeeper: No face? No mask.
        if not self.has_face(img_rgb):
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        results = self.segmenter.process(img_rgb)
        
        if results.segmentation_mask is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Create writable copy
        mask = results.segmentation_mask.copy()

        # Hard Threshold
        mask[mask < 0.5] = 0
        
        # Area Check
        img_area = image.shape[0] * image.shape[1]
        person_area = np.count_nonzero(mask)
        
        if person_area < (img_area * 0.005): 
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            
        # Smoothing
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask

    def blend_human_safe(self, original, corrected_brand, mask):
        person_look = cv2.addWeighted(original, 0.7, corrected_brand, 0.3, 0)
        mask_3d = np.dstack((mask, mask, mask))
        final = (person_look.astype(float) * mask_3d) + \
                (corrected_brand.astype(float) * (1.0 - mask_3d))
        return final.astype("uint8")

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def load_local_references(folder_path="references"):
    images = {}
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return images

    for ext in valid_extensions:
        search_path = os.path.join(folder_path, ext)
        for file_path in glob.glob(search_path):
            try:
                img = Image.open(file_path)
                img_array = np.array(img.convert('RGB'))
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_cv = cv2.resize(img_cv, (300, 300))
                
                filename = os.path.basename(file_path)
                images[filename] = img_cv
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    return images

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==========================================
# 4. USER INTERFACE (STREAMLIT)
# ==========================================

st.set_page_config(page_title="Image Editor ABB", layout="wide")

# --- UI HEADER UPDATED ---
st.title("Image Editor ABB")
st.subheader("Automated Brand Color Corrector")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("Settings")

# Toggles
use_ai = st.sidebar.checkbox("✅ AI Skin Protection", value=True, help="Only active if a face is detected.")
use_contrast = st.sidebar.checkbox("✅ Auto-Contrast Recovery", value=True, help="Stretches histogram to prevent 'washed out' look on cold images.")

st.sidebar.divider()
st.sidebar.subheader("Reference Library")

# Load references
reference_images = load_local_references("references")

if reference_images:
    st.sidebar.success(f"{len(reference_images)} Reference Images Loaded.")
    with st.sidebar.expander("View Active References"):
        for name, img in reference_images.items():
            st.image(bgr_to_rgb(img), caption=name, use_container_width=True)
else:
    st.sidebar.error("No references found!")
    st.sidebar.info("Please create a folder named 'references' and add your brand JPG/PNG files there.")

# --- MAIN AREA ---
# 5 MB limit so uploads work on Render (avoids 400 from proxy/body limits)
target_file = st.file_uploader(
    "Drop image here to process (max 5 MB)",
    type=['png', 'jpg', 'jpeg'],
    max_upload_size=5,
)

if target_file and reference_images:
    pil_image = Image.open(target_file)
    input_img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # Init Engines
    color_engine = ColorMatcher()
    ai_engine = HumanDetector() if use_ai else None
    
    with st.spinner('Processing...'):
        # 1. Find Best Reference
        best_ref, best_ref_name = color_engine.find_best_reference(input_img, reference_images)
        
        # 2. Apply Base Correction (With optional Auto-Contrast)
        corrected_base = color_engine.apply_smart_transfer(input_img, best_ref, use_auto_contrast=use_contrast)
        
        final_img = corrected_base
        mask_visualization = None
        
        # 3. AI Processing (If enabled AND face detected)
        if use_ai:
            mask = ai_engine.get_mask(input_img)
            
            if np.max(mask) > 0.1:
                final_img = ai_engine.blend_human_safe(input_img, corrected_base, mask)
                mask_visualization = 1.0 - mask 
            else:
                # No Toast needed for empty mask (silent fail is better for UX here)
                pass
    
    # --- RESULTS DISPLAY ---
    st.success(f"Matched Guideline: **{best_ref_name}**")
    
    if mask_visualization is not None:
        c1, c2, c3 = st.columns(3)
    else:
        c1, c2 = st.columns(2)
        
    with c1:
        st.caption("Original")
        st.image(bgr_to_rgb(input_img), use_container_width=True)
        
    if mask_visualization is not None:
        with c2:
            st.caption("AI Protection Mask")
            st.image(mask_visualization, clamp=True, use_container_width=True)
            
    with (c3 if mask_visualization is not None else c2):
        st.caption("Final Result")
        st.image(bgr_to_rgb(final_img), use_container_width=True)

    # Download
    result_pil = Image.fromarray(bgr_to_rgb(final_img))
    import io
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=95)
    st.download_button("⬇️ Download Image", buf.getvalue(), f"ABB_fixed_{target_file.name}", "image/jpeg")

elif target_file and not reference_images:
    st.warning("⚠️ System halted. Please add images to the 'references' folder.")