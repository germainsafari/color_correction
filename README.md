---
title: Color Correction API
emoji: 🎨
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Color Correction API

FastAPI/Uvicorn microservice for automated brand color correction.  
Upload an image and receive a color-graded version matching your brand reference library.

## API Endpoints

### `GET /api/health`

Health check — returns reference count and AI engine status.

```json
{
  "status": "ok",
  "references_loaded": 82,
  "ai_available": true,
  "ai_error": null
}
```

### `POST /api/grade`

Upload an image and receive the graded JPEG.

| Parameter      | Type   | Default | Description                       |
|---------------|--------|---------|-----------------------------------|
| `file`        | File   | —       | Image file (JPEG/PNG, max 20 MB)  |
| `use_ai`      | bool   | `true`  | Enable AI skin protection         |
| `use_contrast` | bool   | `true`  | Enable auto-contrast recovery     |

**Response**: JPEG image with metadata headers:

| Header                | Example          | Description                     |
|-----------------------|------------------|---------------------------------|
| `X-Matched-Reference` | `hero_shot.jpg`  | Best matching reference file    |
| `X-Match-Confidence`  | `72.3`           | Match confidence (0–100)        |
| `X-Chroma-Strength`   | `0.75`           | Applied color correction strength|
| `X-AI-Applied`        | `true`           | Whether AI skin protection ran  |

**Example (curl)**:

```bash
curl -X POST "http://localhost:7860/api/grade?use_ai=true&use_contrast=true" \
  -F "file=@photo.jpg" \
  --output graded.jpg
```

**Example (Next.js / fetch)**:

```typescript
const formData = new FormData();
formData.append("file", imageFile);

const res = await fetch(
  `${process.env.NEXT_PUBLIC_GRADING_API_URL}/api/grade?use_ai=true`,
  { method: "POST", body: formData }
);

const blob = await res.blob();
const url = URL.createObjectURL(blob);
```

### `GET /docs`

Interactive OpenAPI (Swagger) documentation.

## Local Development

```bash
pip install -r requirements.txt
python app.py
# → http://localhost:7860
# → http://localhost:7860/docs  (Swagger UI)
```

Or with uvicorn directly:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

## Next.js Integration

Add to your Next.js `.env.local`:

```env
# Local development
NEXT_PUBLIC_GRADING_API_URL=http://localhost:7860

# Production (after deploying to Hugging Face)
# NEXT_PUBLIC_GRADING_API_URL=https://<your-space>.hf.space
```

## Deployment (Hugging Face Spaces)

1. Push this repo to a Hugging Face Space with **Docker** SDK.
2. Make sure the `references/` folder contains your brand images.
3. The Space will build from the `Dockerfile` and expose port 7860.

## Reference Images

Place brand reference images (JPG/PNG) in the `references/` folder.  
The engine auto-selects the best structural match for each uploaded image.
