import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import glob

# MediaPipe is imported lazily inside HumanDetector so the app can start and pass
# health checks even when MediaPipe fails (e.g. Python 3.13 or missing libs on Render).

# ==========================================
# 1. COLOR ENGINE (Statistical / Reinhard)
# ==========================================
class ColorMatcher:
    def get_image_stats(self, image):
        # Convert to LAB space (L=Lightness, A=Green/Red, B=Blue/Yellow)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
        (l, a, b) = cv2.split(image_lab)
        return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())

    def find_best_reference(self, source_img, reference_images, ref_match_data=None):
        """
        Content-based reference matching using STRUCTURAL SIMILARITY.
        
        Converts images to grayscale and uses histogram equalization + 
        Normalized Cross-Correlation (NCC) to match by scene structure,
        completely independent of color grading differences.
        
        Args:
            source_img: Input BGR image.
            reference_images: Dict of {name: BGR image}.
            ref_match_data: Optional dict of {name: precomputed float32
                            equalized grayscale}. Skips redundant per-call
                            conversion of all 82+ references.
        """
        match_size = (300, 300)
        
        # Prepare source: grayscale → equalize → float (only the INPUT, once)
        src_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        src_resized = cv2.resize(src_gray, match_size)
        src_eq = cv2.equalizeHist(src_resized).astype(np.float32)
        
        best_ref = None
        best_score = -float('inf')
        best_ref_name = ""
        
        for name, ref_img in reference_images.items():
            # Use precomputed data when available (cached path — fast)
            if ref_match_data and name in ref_match_data:
                ref_eq = ref_match_data[name]
            else:
                # Fallback: compute on the fly (slow path)
                ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                ref_resized = cv2.resize(ref_gray, match_size)
                ref_eq = cv2.equalizeHist(ref_resized).astype(np.float32)
            
            # NCC on equalized grayscale: compares structural patterns
            # (edges, shapes, textures) independent of color/brightness
            ncc = cv2.matchTemplate(
                src_eq, ref_eq, cv2.TM_CCOEFF_NORMED
            )[0][0]
            
            if ncc > best_score:
                best_score = ncc
                best_ref = ref_img
                best_ref_name = name
                
        return best_ref, best_ref_name, best_score

    def apply_smart_transfer(self, source, target, use_auto_contrast=True, chroma_strength=0.75):
        """
        Applies color correction with configurable chroma intensity.
        
        Args:
            source: Input BGR image to correct.
            target: Reference BGR image (the desired look).
            use_auto_contrast: If True, performs dynamic range stretching.
            chroma_strength: 0.0 = keep original colors, 1.0 = full Reinhard.
                             Default 0.75 gives strong correction while
                             retaining some original tonality to prevent
                             extreme yellow/blue casts.
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
        
        # 1. Color (Chroma - A/B Channels): Blended Reinhard Transfer
        #    Full Reinhard can push colors too aggressively (too yellow/blue).
        #    Blending retains some original tonality for a more natural result.
        a_reinhard = ((a_src - a_mean_src) * (a_std_tar / (a_std_src + eps))) + a_mean_tar
        b_reinhard = ((b_src - b_mean_src) * (b_std_tar / (b_std_src + eps))) + b_mean_tar
        
        a_new = a_src * (1.0 - chroma_strength) + a_reinhard * chroma_strength
        b_new = b_src * (1.0 - chroma_strength) + b_reinhard * chroma_strength

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
        """
        Lazily import and initialize MediaPipe.
        Any import / runtime errors should be handled by the caller.
        """
        import mediapipe as mp
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

@st.cache_resource(show_spinner="Loading reference library...")
def load_local_references(folder_path="references"):
    """
    Loads reference images AND precomputes grayscale-equalized matching data.
    Cached so the 82+ images are only loaded & processed once per app session,
    not on every user interaction.
    """
    images = {}
    match_data = {}  # Pre-computed matching data (grayscale equalized float32)
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    match_size = (300, 300)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return images, match_data

    for ext in valid_extensions:
        search_path = os.path.join(folder_path, ext)
        for file_path in glob.glob(search_path):
            try:
                img = Image.open(file_path)
                img_array = np.array(img.convert('RGB'))
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_cv = cv2.resize(img_cv, match_size)
                
                filename = os.path.basename(file_path)
                images[filename] = img_cv
                
                # Pre-compute equalized grayscale for fast structural matching
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                gray_eq = cv2.equalizeHist(gray).astype(np.float32)
                match_data[filename] = gray_eq
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
    return images, match_data

@st.cache_resource(show_spinner=False)
def get_ai_engine():
    """Cache the MediaPipe models so they are loaded once, not on every rerun."""
    try:
        return HumanDetector(), None
    except Exception as e:
        return None, str(e)

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

# Auto-adjusted chroma: 0.75 for confident matches, 0.25 for low confidence.
# Stored in session_state so the slider visually reflects the active value.
if 'effective_chroma' not in st.session_state:
    st.session_state.effective_chroma = 0.75

st.sidebar.slider(
    "Color Correction Strength",
    min_value=0.0, max_value=1.0,
    value=st.session_state.effective_chroma,
    step=0.05,
    disabled=True,
    help="Automatically set: 0.75 for matches ≥50% confidence, "
         "0.25 for matches <50% confidence."
)

st.sidebar.divider()
st.sidebar.subheader("Reference Library")

# Load references (cached — only loaded once per app session)
reference_images, ref_match_data = load_local_references("references")

if reference_images:
    st.sidebar.success(f"{len(reference_images)} Reference Images Loaded.")
    with st.sidebar.expander("View Active References"):
        for name, img in reference_images.items():
            st.image(bgr_to_rgb(img), caption=name, width='stretch')
else:
    st.sidebar.error("No references found!")
    st.sidebar.info("Please create a folder named 'references' and add your brand JPG/PNG files there.")

# --- MAIN AREA ---
# 20 MB limit; increase server maxUploadSize in .streamlit/config.toml to match
target_file = st.file_uploader(
    "Drop image here to process (max 20 MB)",
    type=['png', 'jpg', 'jpeg'],
    max_upload_size=20,
)

if target_file and reference_images:
    pil_image = Image.open(target_file)
    input_img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # Init Engines (ColorMatcher is lightweight; AI engine is cached)
    color_engine = ColorMatcher()
    ai_engine = None
    if use_ai:
        ai_engine, ai_error = get_ai_engine()
        if ai_engine is None and ai_error:
            st.sidebar.warning("AI face protection unavailable (MediaPipe issue). Color correction only.")
            st.sidebar.caption(f"MediaPipe error: {ai_error}")
    
    with st.spinner('Processing...'):
        # 1. Find Best Reference (content-based structural matching, using precomputed data)
        best_ref, best_ref_name, match_score = color_engine.find_best_reference(
            input_img, reference_images, ref_match_data
        )
        
        # 2. Apply Base Correction (With optional Auto-Contrast + Chroma Strength)
        #    Low-confidence matches (<50%) get reduced chroma strength (0.25)
        #    to avoid pushing colors aggressively when the reference is uncertain.
        new_chroma = 0.25 if match_score < 0.50 else 0.75
        
        # Update sidebar slider if the effective value changed
        if abs(new_chroma - st.session_state.effective_chroma) > 0.01:
            st.session_state.effective_chroma = new_chroma
            st.rerun()
        
        corrected_base = color_engine.apply_smart_transfer(
            input_img, best_ref,
            use_auto_contrast=use_contrast,
            chroma_strength=st.session_state.effective_chroma
        )
        
        final_img = corrected_base
        mask_visualization = None
        
        # 3. AI Processing (If enabled AND face detector is available)
        if use_ai and ai_engine is not None:
            mask = ai_engine.get_mask(input_img)
            
            if np.max(mask) > 0.1:
                final_img = ai_engine.blend_human_safe(input_img, corrected_base, mask)
                mask_visualization = 1.0 - mask 
            else:
                # No Toast needed for empty mask (silent fail is better for UX here)
                pass
    
    # --- RESULTS DISPLAY ---
    confidence_pct = max(0, match_score) * 100
    if match_score >= 0.50:
        st.success(f"Matched Guideline: **{best_ref_name}** (confidence: {confidence_pct:.1f}%)")
    elif match_score >= 0.25:
        st.warning(f"Matched Guideline: **{best_ref_name}** (confidence: {confidence_pct:.1f}% — low match, color strength reduced to 25%)")
    else:
        st.error(f"Matched Guideline: **{best_ref_name}** (confidence: {confidence_pct:.1f}% — no close match, color strength reduced to 25%)")
    
    if mask_visualization is not None:
        c1, c2, c3 = st.columns(3)
    else:
        c1, c2 = st.columns(2)
        
    with c1:
        st.caption("Original")
        st.image(bgr_to_rgb(input_img), width='stretch')
        
    if mask_visualization is not None:
        with c2:
            st.caption("AI Protection Mask")
            st.image(mask_visualization, clamp=True, width='stretch')
            
    with (c3 if mask_visualization is not None else c2):
        st.caption("Final Result")
        st.image(bgr_to_rgb(final_img), width='stretch')

    # Download
    result_pil = Image.fromarray(bgr_to_rgb(final_img))
    import io
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=95)
    st.download_button("⬇️ Download Image", buf.getvalue(), f"ABB_fixed_{target_file.name}", "image/jpeg")

elif target_file and not reference_images:
    st.warning("⚠️ System halted. Please add images to the 'references' folder.")