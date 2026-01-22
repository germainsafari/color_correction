import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import tempfile
import os

# ==========================================
# 1. MOTOR DE COR (Estatística / Reinhard)
# ==========================================
class ColorMatcher:
    def get_image_stats(self, image):
        # Converte para LAB
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
            
            # Peso maior para luminosidade para evitar erros grosseiros de exposição
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
        Aplica a correção de cor preservando 75% do contraste original
        para evitar o efeito 'lavado'.
        """
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        (l_src, a_src, b_src) = cv2.split(source_lab)
        (l_tar, a_tar, b_tar) = cv2.split(target_lab)

        # Stats
        l_mean_src, l_std_src = l_src.mean(), l_src.std()
        a_mean_src, a_std_src = a_src.mean(), a_src.std()
        b_mean_src, b_std_src = b_src.mean(), b_src.std()

        l_mean_tar, l_std_tar = l_tar.mean(), l_tar.std()
        a_mean_tar, a_std_tar = a_tar.mean(), a_tar.std()
        b_mean_tar, b_std_tar = b_tar.mean(), b_tar.std()

        eps = 1e-5
        
        # 1. Cor (A/B): Cópia agressiva da referência
        a_new = ((a_src - a_mean_src) * (a_std_tar / (a_std_src + eps))) + a_mean_tar
        b_new = ((b_src - b_mean_src) * (b_std_tar / (b_std_src + eps))) + b_mean_tar

        # 2. Luz (L): Híbrido (Exposição da Ref + Contraste Original)
        target_mean = l_mean_tar
        # Mistura: 75% Original / 25% Referência
        contrast_blend = (l_std_src * 0.75) + (l_std_tar * 0.25)
        
        l_new = ((l_src - l_mean_src) * (contrast_blend / (l_std_src + eps))) + target_mean

        l_new = np.clip(l_new, 0, 255)
        a_new = np.clip(a_new, 0, 255)
        b_new = np.clip(b_new, 0, 255)

        transfer_lab = cv2.merge([l_new, a_new, b_new])
        transfer_bgr = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
        
        return transfer_bgr

# ==========================================
# 2. MOTOR DE IA (Segmentação de Pessoas)
# ==========================================
class HumanDetector:
    def __init__(self):
        self.mp_selfie = mp.solutions.selfie_segmentation
        # model_selection=1 é 'landscape' (mais lento mas melhor qualidade)
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

    def get_mask(self, image):
        # MediaPipe quer RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(img_rgb)
        
        mask = results.segmentation_mask
        if mask is None:
            # Se der erro, retorna máscara vazia (tudo preto)
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            
        # Suaviza bordas para não ficar serrilhado (Soft Edge)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        return mask

    def blend_human_safe(self, original, corrected_brand, mask):
        """
        Mistura inteligente:
        - Fundo: 100% Brand Look
        - Pessoa: 70% Original / 30% Brand Look (para não destoar)
        """
        # Cria a versão 'Pessoa' (levemente tratada)
        person_look = cv2.addWeighted(original, 0.7, corrected_brand, 0.3, 0)
        
        # Expande máscara para 3 canais para poder multiplicar
        mask_3d = np.dstack((mask, mask, mask))
        
        # Fórmula: (Pessoa * mask) + (Fundo * (1-mask))
        final = (person_look.astype(float) * mask_3d) + \
                (corrected_brand.astype(float) * (1.0 - mask_3d))
                
        return final.astype("uint8")

# ==========================================
# 3. INTERFACE VISUAL
# ==========================================

def load_image(image_file):
    img = Image.open(image_file)
    img_array = np.array(img.convert('RGB'))
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

st.set_page_config(page_title="Brand Corrector AI", layout="wide")

st.title("🎨 Corrector V4 (Estatística + IA)")
st.markdown("Automático com proteção inteligente de tons de pele.")

# --- SIDEBAR ---
st.sidebar.header("Configuração")
use_ai = st.sidebar.checkbox("✅ Ativar Proteção de Pele (IA)", value=True, help="Usa IA para detectar pessoas e suavizar a correção no rosto delas.")

st.sidebar.divider()
st.sidebar.subheader("Referências")
ref_files = st.sidebar.file_uploader("Upload Guidelines", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

reference_images = {}
if ref_files:
    for ref_file in ref_files:
        img = load_image(ref_file)
        img = cv2.resize(img, (300, 300)) 
        reference_images[ref_file.name] = img
    st.sidebar.success(f"{len(reference_images)} referências ativas.")

# --- MAIN AREA ---
target_file = st.file_uploader("Arraste a foto para corrigir", type=['png', 'jpg', 'jpeg'])

if target_file and reference_images:
    input_img = load_image(target_file)
    
    # Inicializa classes
    color_engine = ColorMatcher()
    ai_engine = HumanDetector() if use_ai else None
    
    with st.spinner('Processando imagem...'):
        # 1. Achar melhor referência
        best_ref, best_ref_name = color_engine.find_best_reference(input_img, reference_images)
        
        # 2. Criar Base Corrigida (Agressiva)
        corrected_base = color_engine.apply_smart_transfer(input_img, best_ref)
        
        final_img = corrected_base
        
        # 3. Aplicar IA se ativado
        if use_ai:
            with st.spinner('Detectando humanos (MediaPipe)...'):
                mask = ai_engine.get_mask(input_img)
                # Verifica se encontrou alguém (máscara não é toda preta)
                if np.max(mask) > 0.1:
                    final_img = ai_engine.blend_human_safe(input_img, corrected_base, mask)
                    st.toast("Pessoa detectada e protegida!", icon="👤")
                else:
                    st.toast("Nenhuma pessoa detectada. Aplicando correção total.", icon="info")
    
    # Exibição
    st.success(f"Baseado na referência: **{best_ref_name}**")
    
    c1, c2 = st.columns(2)
    with c1:
        st.image(bgr_to_rgb(input_img), caption="Original", use_container_width=True)
    with c2:
        st.image(bgr_to_rgb(final_img), caption="Resultado Final (Com IA)", use_container_width=True)

    # Download
    result_pil = Image.fromarray(bgr_to_rgb(final_img))
    import io
    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=95)
    st.download_button("⬇️ Baixar Imagem", buf.getvalue(), f"smart_fixed_{target_file.name}", "image/jpeg")

elif target_file and not reference_images:
    st.warning("⚠️ Carregue as imagens de referência na barra lateral.")