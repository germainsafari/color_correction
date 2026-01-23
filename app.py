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

    def apply_smart_transfer(self, source, target):
        """
        Applies color correction and finalizes with a smart 'Auto-Contrast' 
        step to prevent the 'washed-out' look (especially in cold tones).
        """
        # Convert to LAB space (L=Lightness, A=Green/Red, B=Blue/Yellow)
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        (l_src, a_src, b_src) = cv2.split(source_lab)
        (l_tar, a_tar, b_tar) = cv2.split(target_lab)

        # Calculate Statistics (Mean & Standard Deviation)
        l_mean_src, l_std_src = l_src.mean(), l_src.std()
        a_mean_src, a_std_src = a_src.mean(), a_src.std()
        b_mean_src, b_std_src = b_src.mean(), b_src.std()

        l_mean_tar, l_std_tar = l_tar.mean(), l_tar.std()
        a_mean_tar, a_std_tar = a_tar.mean(), a_tar.std()
        b_mean_tar, b_std_tar = b_tar.mean(), b_tar.std()

        eps = 1e-5
        
        # 1. Color (Chroma - A/B Channels): Aggressive match
        # We want to force the brand colors onto the image
        a_new = ((a_src - a_mean_src) * (a_std_tar / (a_std_src + eps))) + a_mean_tar
        b_new = ((b_src - b_mean_src) * (b_std_tar / (b_std_src + eps))) + b_mean_tar

        # 2. Lightness (Luma - L Channel): Soft Transfer
        # We blend the contrast (Std Dev) to avoid destroying the image structure.
        # 80% Original Contrast / 20% Reference Contrast
        contrast_blend = (l_std_src * 0.80) + (l_std_tar * 0.20)
        
        # Apply Reinhard Transfer for Lightness
        l_new = ((l_src - l_mean_src) * (contrast_blend / (l_std_src + eps))) + l_mean_tar

        # --- STEP 3: DYNAMIC RANGE RECOVERY (AUTO-LEVELS) ---
        # Cold photos tend to look gray/flat after transfer. 
        # This step stretches the histogram to ensure true blacks.
        
        # Get the darkest (1%) and brightest (99%) pixel values
        # We use 1% and 99% instead of 0/100 to ignore outliers/dead pixels
        min_val = np.percentile(l_new, 1)
        max_val = np.percentile(l_new, 99)
        
        # Min-Max Normalization (Stretching logic)
        scale = 255.0 / (max_val - min_val + eps)
        l_stretched = (l_new - min_val) * scale
        
        # Final Blend:
        # We don't apply the stretch 100%, as it might look artificial.
        # We mix 30% Stretched result with 70% Reinhard result.
        l_final = (l_stretched * 0.3) + (l_new * 0.7)
        # ----------------------------------------------------

        # Final Clipping to valid 0-255 range
        l_final = np.clip(l_final, 0, 255)
        a_new = np.clip(a_new, 0, 255)
        b_new = np.clip(b_new, 0, 255)

        # Merge channels and convert back to BGR
        transfer_lab = cv2.merge([l_final, a_new, b_new])
        transfer_bgr = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
        
        return transfer_bgr

# ==========================================
# 2. AI ENGINE (Human Segmentation - Fixed)
# ==========================================
class HumanDetector:
    def __init__(self):
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

    def get_mask(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(img_rgb)
        
        # Se não achou nada, retorna preto
        if results.segmentation_mask is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # --- CORREÇÃO DO ERRO AQUI ---
        # O MediaPipe entrega a máscara como "Read-Only". 
        # Usamos .copy() para criar uma versão editável na memória.
        mask = results.segmentation_mask.copy()
        # -----------------------------

        # 1. Threshold Rígido: Limpa ruído (sombras fracas)
        # Zera tudo que a IA não tem pelo menos 50% de certeza
        mask[mask < 0.5] = 0
        
        # 2. Verificação de Área: Ignora "manchas" pequenas
        img_area = image.shape[0] * image.shape[1]
        person_area = np.count_nonzero(mask)
        
        # Se a pessoa ocupa menos de 0.5% da foto, consideramos erro
        if person_area < (img_area * 0.005): 
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            
        # 3. Suavização das bordas
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
    """Loads all images from the specified local folder."""
    images = {}
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) # Create if doesn't exist to avoid errors
        return images

    for ext in valid_extensions:
        search_path = os.path.join(folder_path, ext)
        for file_path in glob.glob(search_path):
            try:
                img = Image.open(file_path)
                img_array = np.array(img.convert('RGB'))
                # Convert RGB (PIL) to BGR (OpenCV)
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                # Resize for performance (stat calculation only)
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

st.set_page_config(page_title="Brand Color Corrector", layout="wide")

st.title("🎨 Automated Brand Color Corrector")
st.markdown("Automatic color grading based on Brand Guidelines with AI Skin Tone Protection.")

# --- SIDEBAR ---
st.sidebar.header("Settings")
use_ai = st.sidebar.checkbox("✅ AI Skin Protection", value=True, help="Detects humans to preserve natural skin tones.")

st.sidebar.divider()
st.sidebar.subheader("Reference Library")

# Load references automatically
reference_images = load_local_references("references")

if reference_images:
    st.sidebar.success(f"{len(reference_images)} Reference Images Loaded.")
    # Optional: Show thumbnails in sidebar
    with st.sidebar.expander("View Active References"):
        for name, img in reference_images.items():
            st.image(bgr_to_rgb(img), caption=name, use_container_width=True)
else:
    st.sidebar.error("No references found!")
    st.sidebar.info("Please create a folder named 'references' and add your brand JPG/PNG files there.")

# --- MAIN AREA ---
target_file = st.file_uploader("Drop image here to process", type=['png', 'jpg', 'jpeg'])

if target_file and reference_images:
    # Load Input
    pil_image = Image.open(target_file)
    input_img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # Init Engines
    color_engine = ColorMatcher()
    ai_engine = HumanDetector() if use_ai else None
    
    with st.spinner('Processing...'):
        # 1. Find Best Reference
        best_ref, best_ref_name = color_engine.find_best_reference(input_img, reference_images)
        
        # 2. Apply Base Correction (Aggressive)
        corrected_base = color_engine.apply_smart_transfer(input_img, best_ref)
        
        final_img = corrected_base
        mask_visualization = None
        
        # 3. AI Processing
        if use_ai:
            mask = ai_engine.get_mask(input_img)
            
            # Check if mask is not empty
            if np.max(mask) > 0.1:
                final_img = ai_engine.blend_human_safe(input_img, corrected_base, mask)
                
                # Create Visualization: White Background, Black Person
                # Mask is 1.0 for person, 0.0 for background.
                # Inverted: 0.0 for person (Black), 1.0 for background (White)
                mask_visualization = 1.0 - mask 
                
            else:
                st.toast("No humans detected. Full correction applied.", icon="ℹ️")
    
    # --- RESULTS DISPLAY ---
    st.success(f"Matched Guideline: **{best_ref_name}**")
    
    # Dynamic columns: 2 or 3 depending on AI usage
    if mask_visualization is not None:
        c1, c2, c3 = st.columns(3)
    else:
        c1, c2 = st.columns(2)
        
    with c1:
        st.subheader("Original")
        st.image(bgr_to_rgb(input_img), use_container_width=True)
        
    if mask_visualization is not None:
        with c2:
            st.subheader("AI Mask")
            # Display grayscale mask. 
            # Clamp allows showing 0-1 floats correctly in Streamlit
            st.image(mask_visualization, caption="Black areas are protected", clamp=True, use_container_width=True)
            
    with (c3 if mask_visualization is not None else c2):
        st.subheader("Final Result")
        st.image(bgr_to_rgb(final_img), use_container_width=True)

    # Download Button
    result_pil = Image.fromarray(bgr_to_rgb(final_img))
    import io
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=95)
    st.download_button("⬇️ Download Image", buf.getvalue(), f"brand_fixed_{target_file.name}", "image/jpeg")

elif target_file and not reference_images:
    st.warning("⚠️ System halted. Please add images to the 'references' folder.")