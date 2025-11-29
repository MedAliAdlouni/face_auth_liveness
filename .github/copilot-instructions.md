# AI Coding Agent Instructions for Face Authentication & Liveness Detection

## Project Overview
A Python-based biometric authentication system combining face recognition with anti-spoofing (liveness detection). The system performs three-step verification: face detection → liveness verification → face embedding matching.

## Architecture & Data Flow

### Core Pipeline Architecture
```
User Input (image/webcam)
    ↓
[Face Detection] (MTCNN)
    ↓
[Liveness Detection] (LivenessDetector)
    ↓
[Face Embedding] (InceptionResnetV1)
    ↓
[Embedding Comparison] (Cosine similarity)
    ↓
Authentication Result
```

### Two Main Pipelines
- **Registration** (`src/pipeline/registration_pipeline.py`): Saves face embedding to `data/embeddings/{first_name}_{last_name}.pkl`
- **Authentication** (`src/pipeline/authentication_pipeline.py`): Verifies user liveness + compares embeddings against stored file

### Key Components
- **Face Detection**: `src/detectors/face_detector.py` uses `facenet_pytorch.MTCNN` (160×160 crops, margin=14)
- **Embedding Extraction**: `src/embedding/face_embedding.py` uses `InceptionResnetV1` (VGGFace2 pretrained)
- **Liveness Detection**: `src/liveness/liveness_detector.py` implements multi-metric spoof detection
- **Helper Functions**: `src/utils/helpers.py` manages embeddings storage, image I/O, and pipeline orchestration

## Liveness Detection Algorithm (Critical)

The `LivenessDetector` class uses **4 concurrent detection methods** (scored 0-9 points, require ≥5 for "real"):

| Method | Implementation | Score | Threshold |
|--------|---|---|---|
| **Texture Analysis** | Laplacian variance + Sobel edge detection | +2 | >25 (textured skin) |
| **Motion Detection** | Face coordinate variance over 30-frame history | +2 | 2 < variance < 80 (natural movement) |
| **Color Variation** | HSV/YCrCb channel analysis (blood flow simulation) | +1 | consistency < 50 |
| **Screen Detection** | FFT spectrum + brightness uniformity | +1 (-2 penalty) | std < 30 & spectrum peaks < 10k |

**Final Verdict**: Real if `(criteria_passed ≥ 2) AND (real_count/total_count ≥ 60%)`

## Critical Parameters & Thresholds

```python
# In src/utils/helpers.py
EMBEDDINGS_DIR = "data/embeddings"  # User embedding storage
compare_embeddings() similarity_threshold = 0.70  # Cosine similarity (LOWERED from 0.85)

# In src/liveness/liveness_detector.py
texture_threshold = 25  # Increased for stricter fake detection
motion_history = deque(maxlen=30)  # 30-frame motion window
```

## Embedding Storage Format

**File**: `data/embeddings/{firstname_lastname}.pkl` (pickle format)
```python
{
    'first_name': str,
    'last_name': str,
    'embedding': torch.Tensor (shape: 1×512),
    'registration_date': timestamp_string
}
```

**Key Design**: Embeddings are 512-dim PyTorch tensors from InceptionResnetV1. Always `.squeeze()` before cosine similarity.

## Common Patterns & Code Conventions

### Image Input Pattern
```python
# Flexible input: PIL Image, np.ndarray, or webcam capture
from src.utils.helpers import upload_image
img = upload_image()  # Returns PIL.Image

# Webcam: countdown timer, conversion to RGB
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
```

### Embedding Management
```python
# Load: Returns torch.Tensor or None if not found
embedding = load_user_embedding(first_name, last_name)

# Compare: Returns (cosine_sim_float, bool_match)
similarity, is_match = compare_embeddings(emb1, emb2, threshold=0.70)

# Generate path: Always lowercase filenames
path = get_embedding_path("John", "Doe")  # → "data/embeddings/john_doe.pkl"
```

### Pipeline Return Signatures
Both pipelines return dicts (never raise exceptions for business logic failures):
```python
# Success: {'success': True, 'user': name, ...metrics...}
# Failure: {'success': False, 'message': error_text, 'user': name}
```

## Device Handling
- **GPU Priority**: Code checks `torch.cuda.is_available()` in `extract_face_embedding()`
- **Webcam**: Always check `cap.isOpened()` after `cv2.VideoCapture(0)`
- **Windows-Specific**: Uses `cv2.imshow()` for visualization (not headless-compatible)

## Pre-Existing Hardcoded Values (Do Not Change Without Testing)
- Face detection confidence threshold: `0.5` (in `registration_pipeline.py`)
- Liveness duration: `10` seconds (default parameter)
- Default similarity threshold: `0.65` (lowered from `0.85` per comment)
- OpenCV Haar cascade: `haarcascade_frontalface_default.xml` (not deep learning-based)

## Dependencies
- **ML Models**: `facenet_pytorch` (MTCNN, InceptionResnetV1)
- **Computer Vision**: `cv2` (Haar cascades, FFT, Sobel/Laplacian operators)
- **Core**: `torch`, `numpy`, `PIL`, `pickle`
- **Utility**: `time`, `deque`, `os`

## Testing & Debugging Notes
- Liveness detection sensitive to **lighting conditions** (affects texture/color analysis)
- Motion threshold `2-80` designed to reject static images AND high-frequency noise
- Screen detection uses FFT—may trigger false positives on patterned backgrounds
- Texture threshold `>25` filters flat/low-quality captures
- Embeddings are normalized but **not batch-friendly** by default (requires `.unsqueeze()`)

## Anti-Patterns to Avoid
1. ❌ Don't instantiate MTCNN/InceptionResnetV1 multiple times per call (expensive)
2. ❌ Don't skip `.squeeze()` on embeddings before cosine similarity
3. ❌ Don't hardcode paths—use `get_embedding_path()` function
4. ❌ Don't modify thresholds without re-testing on representative spoof/real data
5. ❌ Don't assume PIL Image in embedding code—MTCNN accepts multiple formats
