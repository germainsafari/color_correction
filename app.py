import io
import os
import glob
import logging
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("color_correction")

# MediaPipe is imported lazily inside HumanDetector so the app can start and
# pass health checks even when MediaPipe fails (e.g. Python 3.13 or missing
# native libs).

# ==========================================
# 1. COLOR ENGINE (Statistical / Reinhard)
# ==========================================
class ColorMatcher:
    """Reinhard-based color transfer in CIE-LAB space."""

    def get_image_stats(self, image: np.ndarray):
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
        (l, a, b) = cv2.split(image_lab)
        return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())

    def find_best_reference(
        self,
        source_img: np.ndarray,
        reference_images: dict[str, np.ndarray],
        ref_match_data: dict[str, np.ndarray] | None = None,
    ):
        """
        Content-based reference matching using STRUCTURAL SIMILARITY.

        Converts images to grayscale and uses histogram equalization +
        Normalized Cross-Correlation (NCC) to match by scene structure,
        completely independent of color grading differences.
        """
        match_size = (300, 300)

        src_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        src_resized = cv2.resize(src_gray, match_size)
        src_eq = cv2.equalizeHist(src_resized).astype(np.float32)

        best_ref = None
        best_score = -float("inf")
        best_ref_name = ""

        for name, ref_img in reference_images.items():
            if ref_match_data and name in ref_match_data:
                ref_eq = ref_match_data[name]
            else:
                ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                ref_resized = cv2.resize(ref_gray, match_size)
                ref_eq = cv2.equalizeHist(ref_resized).astype(np.float32)

            ncc = cv2.matchTemplate(src_eq, ref_eq, cv2.TM_CCOEFF_NORMED)[0][0]

            if ncc > best_score:
                best_score = ncc
                best_ref = ref_img
                best_ref_name = name

        return best_ref, best_ref_name, best_score

    def apply_smart_transfer(
        self,
        source: np.ndarray,
        target: np.ndarray,
        use_auto_contrast: bool = True,
        chroma_strength: float = 0.75,
    ) -> np.ndarray:
        """
        Applies color correction with configurable chroma intensity.

        Args:
            source: Input BGR image to correct.
            target: Reference BGR image (the desired look).
            use_auto_contrast: Performs dynamic range stretching.
            chroma_strength: 0.0 = keep original, 1.0 = full Reinhard.
        """
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        (l_src, a_src, b_src) = cv2.split(source_lab)
        (l_tar, a_tar, b_tar) = cv2.split(target_lab)

        l_mean_src, l_std_src = l_src.mean(), l_src.std()
        a_mean_src, a_std_src = a_src.mean(), a_src.std()
        b_mean_src, b_std_src = b_src.mean(), b_src.std()

        l_mean_tar, l_std_tar = l_tar.mean(), l_tar.std()
        a_mean_tar, a_std_tar = a_tar.mean(), a_tar.std()
        b_mean_tar, b_std_tar = b_tar.mean(), b_tar.std()

        eps = 1e-5

        # Chroma (A/B channels): blended Reinhard transfer
        a_reinhard = ((a_src - a_mean_src) * (a_std_tar / (a_std_src + eps))) + a_mean_tar
        b_reinhard = ((b_src - b_mean_src) * (b_std_tar / (b_std_src + eps))) + b_mean_tar

        a_new = a_src * (1.0 - chroma_strength) + a_reinhard * chroma_strength
        b_new = b_src * (1.0 - chroma_strength) + b_reinhard * chroma_strength

        # Lightness (L channel): soft transfer (80 % original / 20 % reference)
        contrast_blend = (l_std_src * 0.80) + (l_std_tar * 0.20)
        l_new = ((l_src - l_mean_src) * (contrast_blend / (l_std_src + eps))) + l_mean_tar

        l_final = l_new
        if use_auto_contrast:
            min_val = np.percentile(l_new, 1)
            max_val = np.percentile(l_new, 99)
            scale = 255.0 / (max_val - min_val + eps)
            l_stretched = (l_new - min_val) * scale
            l_final = (l_stretched * 0.3) + (l_new * 0.7)

        l_final = np.clip(l_final, 0, 255)
        a_new = np.clip(a_new, 0, 255)
        b_new = np.clip(b_new, 0, 255)

        transfer_lab = cv2.merge([l_final, a_new, b_new])
        return cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2BGR)


# ==========================================
# 2. AI ENGINE (Face Detection + Segmentation)
# ==========================================
class HumanDetector:
    def __init__(self):
        import mediapipe as mp

        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=0.6
        )
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

    def has_face(self, image_rgb: np.ndarray) -> bool:
        results = self.face_detector.process(image_rgb)
        return results.detections is not None

    def get_mask(self, image: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.has_face(img_rgb):
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        results = self.segmenter.process(img_rgb)
        if results.segmentation_mask is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        mask = results.segmentation_mask.copy()
        mask[mask < 0.5] = 0

        img_area = image.shape[0] * image.shape[1]
        person_area = np.count_nonzero(mask)
        if person_area < (img_area * 0.005):
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask

    def blend_human_safe(
        self,
        original: np.ndarray,
        corrected_brand: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        person_look = cv2.addWeighted(original, 0.7, corrected_brand, 0.3, 0)
        mask_3d = np.dstack((mask, mask, mask))
        final = (person_look.astype(float) * mask_3d) + (
            corrected_brand.astype(float) * (1.0 - mask_3d)
        )
        return final.astype("uint8")


# ==========================================
# 3. REFERENCE LOADER
# ==========================================
def load_local_references(folder_path: str = "references"):
    """Load reference images and precompute grayscale-equalized matching data."""
    images: dict[str, np.ndarray] = {}
    match_data: dict[str, np.ndarray] = {}
    valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    match_size = (300, 300)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return images, match_data

    for ext in valid_extensions:
        search_path = os.path.join(folder_path, ext)
        for file_path in glob.glob(search_path):
            try:
                img = Image.open(file_path)
                img_array = np.array(img.convert("RGB"))
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_cv = cv2.resize(img_cv, match_size)

                filename = os.path.basename(file_path)
                images[filename] = img_cv

                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                gray_eq = cv2.equalizeHist(gray).astype(np.float32)
                match_data[filename] = gray_eq
            except Exception as e:
                logger.warning("Error loading %s: %s", file_path, e)

    return images, match_data


# ==========================================
# 4. APPLICATION STATE (loaded once at startup)
# ==========================================
class AppState:
    """Holds pre-loaded models and reference data."""

    reference_images: dict[str, np.ndarray] = {}
    ref_match_data: dict[str, np.ndarray] = {}
    color_engine: ColorMatcher = ColorMatcher()
    ai_engine: HumanDetector | None = None
    ai_error: str | None = None


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load heavy resources once when the server starts."""
    logger.info("Loading reference library...")
    state.reference_images, state.ref_match_data = load_local_references("references")
    logger.info("Loaded %d reference images.", len(state.reference_images))

    logger.info("Initializing AI engine (MediaPipe)...")
    try:
        state.ai_engine = HumanDetector()
        logger.info("AI engine ready.")
    except Exception as e:
        state.ai_error = str(e)
        logger.warning("AI engine unavailable: %s", e)

    yield  # app is running

    logger.info("Shutting down.")


# ==========================================
# 5. FASTAPI APPLICATION
# ==========================================
app = FastAPI(
    title="Color Correction API",
    description="Automated brand color correction microservice. "
    "Upload an image and receive a color-graded version.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow any origin so your Next.js app can call from localhost or production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
}


def _read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR numpy array."""
    pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _encode_jpeg(image_bgr: np.ndarray, quality: int = 95) -> bytes:
    """Encode a BGR image to JPEG bytes."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "Color Correction API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "grade": "POST /api/grade",
            "health": "GET /api/health",
        },
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "references_loaded": len(state.reference_images),
        "ai_available": state.ai_engine is not None,
        "ai_error": state.ai_error,
    }


@app.post("/api/grade")
async def grade_image(
    file: UploadFile = File(..., description="Image file (JPEG/PNG, max 20 MB)"),
    use_ai: bool = Query(True, description="Enable AI skin protection"),
    use_contrast: bool = Query(True, description="Enable auto-contrast recovery"),
):
    """
    Upload an image and receive the color-graded version as a JPEG response.

    The service automatically selects the best brand reference and applies
    Reinhard color transfer.  When a face is detected and ``use_ai`` is True,
    skin tones are preserved via MediaPipe segmentation blending.
    """
    # --- validation ---------------------------------------------------------
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
            f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 20 MB limit.")

    if not state.reference_images:
        raise HTTPException(
            status_code=503,
            detail="No reference images loaded. The service is not ready.",
        )

    # --- decode -------------------------------------------------------------
    try:
        input_img = _read_image_from_upload(file_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # --- processing ---------------------------------------------------------
    engine = state.color_engine

    # 1. Find best reference
    best_ref, best_ref_name, match_score = engine.find_best_reference(
        input_img, state.reference_images, state.ref_match_data
    )

    # 2. Adaptive chroma strength
    chroma_strength = 0.25 if match_score < 0.50 else 0.75

    # 3. Apply color correction
    corrected = engine.apply_smart_transfer(
        input_img,
        best_ref,
        use_auto_contrast=use_contrast,
        chroma_strength=chroma_strength,
    )

    final_img = corrected

    # 4. AI skin protection (if enabled and available)
    ai_applied = False
    if use_ai and state.ai_engine is not None:
        mask = state.ai_engine.get_mask(input_img)
        if np.max(mask) > 0.1:
            final_img = state.ai_engine.blend_human_safe(input_img, corrected, mask)
            ai_applied = True

    # --- encode & respond ---------------------------------------------------
    jpeg_bytes = _encode_jpeg(final_img)
    confidence_pct = max(0.0, float(match_score)) * 100

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "X-Matched-Reference": best_ref_name,
            "X-Match-Confidence": f"{confidence_pct:.1f}",
            "X-Chroma-Strength": f"{chroma_strength:.2f}",
            "X-AI-Applied": str(ai_applied).lower(),
            "Content-Disposition": f'inline; filename="graded_{file.filename}"',
        },
    )


# ==========================================
# 6. ENTRY POINT
# ==========================================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
