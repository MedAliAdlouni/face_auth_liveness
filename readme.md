
# Face Authentication & Liveness Detection

Comprehensive Python-based biometric authentication combining face recognition with multi-metric liveness (anti-spoofing) checks. The project performs three-step verification: face detection → liveness verification → face embedding matching.

**Table of Contents**
- **Project Overview**: What this project does and core features
- **Architecture**: Key components and dataflow
- **Quick Start**: Install, run, and example commands (Windows PowerShell)
- **Usage**: Registration and authentication pipelines
- **Liveness Detection**: Algorithm summary and thresholds
- **Embeddings**: Storage format and comparison rules
- **Testing**: Running unit tests
- **Development Notes**: Important conventions and anti-patterns
- **Contributing & License**

**Project Overview**
- **Purpose**: Provide a secure face-based authentication pipeline that resists common spoofing attempts using multi-metric liveness detection.
- **Languages & Libraries**: Python, `torch`, `facenet_pytorch`, `opencv` (`cv2`), `PIL`, `numpy`, `pickle`.

**Architecture**
The system pipeline:

User Input (image/webcam)
		↓
[Face Detection] (`src/detectors/face_detector.py` using `facenet_pytorch.MTCNN`)
		↓
[Liveness Detection] (`src/liveness/liveness_detector.py`)
		↓
[Face Embedding] (`src/embedding/face_embedding.py` using `InceptionResnetV1`)
		↓
[Embedding Comparison] (`src/utils/embeddings.py`) → Authentication Result

Two main CLI pipelines:
- **Registration**: `src/pipeline/registration_pipeline.py` — extract and save user embedding
- **Authentication**: `src/pipeline/authentication_pipeline.py` — run liveness checks then compare embedding to stored user

**Quick Start (Windows PowerShell)**
- **Create / Activate venv**:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
- **Install requirements**:
```
pip install -r requirements.txt
```
- **Run registration (example)**:
```
python -m src.pipeline.registration_pipeline
```
- **Run authentication (example)**:
```
python -m src.pipeline.authentication_pipeline
```
Note: The repository includes a prebuilt virtual environment under `face/` — if you prefer, use `face\Scripts\python.exe` the same way shown in project examples.

**Usage — Pipelines**
- **Registration pipeline** (`src/pipeline/registration_pipeline.py`):
	- Detects a face, extracts a 512-d embedding, and stores it in `data/embeddings/{firstname_lastname}.pkl`.
	- Default face detection confidence threshold: `0.5`.
- **Authentication pipeline** (`src/pipeline/authentication_pipeline.py`):
	- Captures image or webcam frames, performs liveness checks, then compares the resulting embedding against stored embeddings.
	- Returns a standardized dict: `{'success': True/False, ...}`

**Liveness Detection (Summary)**
Liveness checks are implemented in `src/liveness/liveness_detector.py` and combine multiple metrics to decide whether a face is "real" or a spoof. The detector runs several concurrent checks (each scored) and applies thresholds to reach a final verdict.

- **Texture Analysis**: Laplacian variance + Sobel edge detection — assigns +2 if textured skin detected. Default threshold: `> 25`.
- **Motion Detection**: Variance of face coordinates across a frame history (default history length: 30 frames) — assigns +2 when `2 < variance < 80`.
- **Color Variation**: HSV / YCrCb channel analysis for subtle blood-flow/color cues — assigns +1 when consistency < 50.
- **Screen Detection**: FFT spectrum analysis + brightness uniformity — assigns +1 (with possible -2 penalty) when uniform/low-frequency patterns detected (std < 30 and spectrum peaks < 10000).

Final liveness verdict: a face is considered *real* if both of these hold: `(criteria_passed >= 2)` AND `(real_count / total_count >= 60%)`.

Important implementation details:
- **Motion window**: `collections.deque(maxlen=30)` used for a 30-frame motion history.
- **Liveness duration**: default `10` seconds for live capture runs.

**Embeddings & Comparison**
- **Storage path**: `data/embeddings/{firstname_lastname}.pkl` (filenames are lowercased by helpers).
- **File format** (pickle) contains a dictionary with keys:
	- `'first_name'`: `str`
	- `'last_name'`: `str`
	- `'embedding'`: `torch.Tensor` (shape `1x512`) — embeddings are produced by `InceptionResnetV1` and should be `.squeeze()` before comparison
	- `'registration_date'`: timestamp string
- **Default similarity threshold**: the codebase references both `0.65` (default) and comparator threshold `0.70` in helpers; check `src/utils/embeddings.py` or `src/utils/helpers.py` for the runtime value. Typical usage: `similarity >= 0.70` for a match.
- **Comparison**: cosine similarity between normalized embeddings. Always call `.squeeze()` on tensors before computing similarity.

**Configuration & Files**
- **Main config**: `src/config.py`
- **Face detector**: `src/detectors/face_detector.py` (uses `facenet_pytorch.MTCNN`, crops at 160×160 with margin)
- **Embedding extractor**: `src/embedding/face_embedding.py` (uses `InceptionResnetV1` pretrained on VGGFace2)
- **Liveness logic**: `src/liveness/liveness_detector.py` and helpers in `src/utils/liveness_utils.py`

**Device Handling & Runtime Notes**
- **GPU priority**: `torch.cuda.is_available()` is used to pick the device in embedding extraction when available.
- **Webcam**: code uses `cv2.VideoCapture(0)` — always check `cap.isOpened()` before reading frames.
- **Display**: `cv2.imshow()` calls are used for local visualization; not headless-friendly.

**Testing**
- Run unit tests with `pytest`:
```
python -m pytest tests
```
or, if using the provided `face` venv:
```
.\face\Scripts\python.exe -m pytest tests
```

**Development Notes & Anti-Patterns**
- **Do not** instantiate `MTCNN` or `InceptionResnetV1` repeatedly per call — expensive. Reuse model instances.
- **Do not** skip `.squeeze()` on embeddings before cosine similarity — shape mismatches cause incorrect results.
- **Do not** hardcode embedding paths — use helper `get_embedding_path()` in `src/utils/embeddings.py`.
- The liveness detector is sensitive to lighting; test thresholds on representative spoof/real data before changing them.

**Contributing**
- Fork the repository, create a feature branch, and open a pull request. Add unit tests for new behavior and run `pytest`.

**License**
- This repository does not contain a license file — add `LICENSE` if you plan to open-source it. Contact the repository owner for license details.

**Contact / Maintainer**
- Repository: `face_auth_liveness` (owner: `MedAliAdlouni`)
- For questions, open an issue in the project or contact the maintainer via the repo.

--
Generated README based on project sources and developer guidance in `.github/copilot-instructions.md`.

