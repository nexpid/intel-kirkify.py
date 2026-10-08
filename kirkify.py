import sys
import os

from subprocess import run
from concurrent.futures import ThreadPoolExecutor, as_completed
from shutil import rmtree
from pathlib import Path
from random import randint
from contextlib import contextmanager
from tqdm import tqdm

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import insightface
from insightface.app import FaceAnalysis
from cv2 import imread, imwrite, VideoCapture, CAP_PROP_FPS
import sys
import os

from subprocess import run
from concurrent.futures import ThreadPoolExecutor, as_completed
from shutil import rmtree
from pathlib import Path
from random import randint
from contextlib import contextmanager
from tqdm import tqdm

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import insightface
from insightface.app import FaceAnalysis
from cv2 import imread, imwrite, VideoCapture, CAP_PROP_FPS

# Constant directories
UNPROCESSED_DIR = Path('unprocessed_frames')
PROCESSED_DIR = Path('processed_frames')
KIRKS_DIR = Path('kirks')

# Utilities for ONNX Runtime/GPU verification
def ort_available_and_providers() -> tuple[bool, list[str]]:
    try:
        import onnxruntime as ort  # type: ignore
        providers = []
        try:
            providers = list(ort.get_available_providers())  # e.g., ['CUDAExecutionProvider','CPUExecutionProvider']
        except Exception:
            providers = []
        if 'CUDAExecutionProvider' in providers:
            try:
                # If directory is not provided, it will auto-search in site-packages/nvidia/...
                ort.preload_dlls()
                print("[INFO] CUDA DLLs preloaded from nvidia-* wheels")
            except Exception as e:
                print("[WARN] Could not preload CUDA DLLs:", e)
        return True, providers
    except Exception:
        return False, []

def get_session_providers(session) -> list[str]:
    try:
        return list(session.get_providers())
    except Exception:
        return []

def get_faceanalysis_providers(faceanalysis: FaceAnalysis) -> dict[str, list[str]]:
    active = {}
    for model_name, model in getattr(faceanalysis, 'models', {}).items():
        session = getattr(model, 'session', None) or getattr(model, 'model', None)
        providers = get_session_providers(session) if session is not None else []
        if providers:
            active[model_name] = providers
    return active

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def initialize_faceanalysis_and_swapper(det_size=(640, 640), ctx_id: int = 0, swapper_providers=None):
    faceanalysis = FaceAnalysis(name="buffalo_l", providers=swapper_providers)
    faceanalysis.prepare(ctx_id=ctx_id, det_size=det_size)
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
    if swapper_providers:
        try:
            sess = getattr(swapper, 'session', None) or getattr(swapper, 'model', None)
            if sess and hasattr(sess, 'set_providers'):
                sess.set_providers(swapper_providers)
        except Exception as exc:
            raise RuntimeError(f"Could not set INSwapper providers: {exc}") from exc
    return faceanalysis, swapper

def get_video_fps(video_path: str) -> float:
    cap = VideoCapture(video_path)
    fps = cap.get(CAP_PROP_FPS)
    cap.release()
    return fps

def extract_frames(video_path: str):
    """Extract all frames to unprocessed_frames/"""
    UNPROCESSED_DIR.mkdir(exist_ok=True)
    run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', video_path,
        '-q:v', '2',
        str(UNPROCESSED_DIR / 'frame_%04d.png')
    ], check=True)

def extract_audio(video_path: str) -> str:
    """Extract audio to audio.aac"""
    run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', video_path,
        '-map', '0:a', '-acodec', 'copy',
        'audio.aac'
    ], check=True)
    return 'audio.aac'

def reconstruct_video(fps: float, audio_path: str, output_path: str, frame_step: int):
    """Combine processed frames with audio"""
    PROCESSED_DIR.mkdir(exist_ok=True)
    run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(fps / frame_step),
        '-i', str(PROCESSED_DIR / 'frame_%04d.png'),
        '-i', audio_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'copy', '-shortest',
        output_path
    ], check=True)

def cleanup(audio_path: str):
    """Remove intermediate files"""
    rmtree(UNPROCESSED_DIR, ignore_errors=True)
    rmtree(PROCESSED_DIR, ignore_errors=True)
    if os.path.exists(audio_path):
        os.remove(audio_path)

def get_random_kirk_face(faceanalysis: FaceAnalysis):
    """Get a random Kirk face, ensuring it has a detected face"""
    for _ in range(3):  # Try up to 3 times
        kirk_path = KIRKS_DIR / f'kirk_{randint(0, 2)}.jpg'
        kirk_img = imread(str(kirk_path))
        faces = faceanalysis.get(kirk_img)
        if faces:
            return faces[0]
    raise RuntimeError("Could not detect a face in any Kirk image")

def kirkify_frame(frame_path: str, output_path: str, faceanalysis: FaceAnalysis, swapper, kirk_face):
    """Process a single frame"""
    img = imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {frame_path}")
    faces = faceanalysis.get(img)

    if faces:  # Only process if faces detected
        res = img.copy()
        for face in faces:
            res = swapper.get(res, face, kirk_face, paste_back=True)
        
        import numpy as np
        if not isinstance(res, np.ndarray):
            res = np.array(res)
        imwrite(output_path, res)
        return True
    else:
        import numpy as np
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        imwrite(output_path, img)
        return False

def process_all_frames(faceanalysis: FaceAnalysis, swapper, frame_step: int = 1, workers: int | None = None) -> dict:
    """Process each frame in unprocessed_frames/ with tqdm progress bar
    - frame_step: process 1 of every N frames (skip to speed up)
    - workers: number of threads for parallel processing
    """
    PROCESSED_DIR.mkdir(exist_ok=True)
    all_files = sorted([f for f in os.listdir(UNPROCESSED_DIR) if f.endswith('.png')])
    frame_files = all_files[::max(1, frame_step)]
    kirk_face = get_random_kirk_face(faceanalysis)

    def _process(idx_and_name: tuple[int, str]) -> bool:
        idx, filename = idx_and_name
        input_path = str(UNPROCESSED_DIR / filename)
        # Renumber output to be contiguous: frame_0000, 0001, ...
        output_path = str(PROCESSED_DIR / f"frame_{idx:04d}.png")
        return kirkify_frame(input_path, output_path, faceanalysis, swapper, kirk_face)

    total = len(frame_files)
    workers = workers or max(1, os.cpu_count() or 1)
    with_faces = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process, (i, fn)) for i, fn in enumerate(frame_files)]
        for fut in tqdm(as_completed(futures), total=total, desc="Processing frames", unit="frame"):
            try:
                if fut.result():
                    with_faces += 1
            except Exception:
                # If a frame fails, count it as no-face and continue
                pass

    return {'total': len(all_files), 'processed': len(frame_files), 'with_faces': with_faces}

def kirkify_video(TARGET_PATH, OUTPUT_PATH, FACE_ANALYSIS, FACE_SWAPPER, frame_step: int = 1, workers: int | None = None):
    print("Extracting frames...")
    extract_frames(TARGET_PATH)
    
    print("Extracting audio...")
    AUDIO_PATH = extract_audio(TARGET_PATH)
    
    print("Processing frames...")
    stats = process_all_frames(FACE_ANALYSIS, FACE_SWAPPER, frame_step=frame_step, workers=workers)
    print(f"  Frames: total {stats['total']}, processed {stats['processed']}, with faces {stats['with_faces']}")
    
    print("Reconstructing video...")
    FPS = get_video_fps(TARGET_PATH)
    reconstruct_video(FPS, AUDIO_PATH, OUTPUT_PATH, frame_step)
    
    print("Cleaning up...")
    cleanup(AUDIO_PATH)
    
    print(f"Done! Output saved to {OUTPUT_PATH}")

def kirkify_image(TARGET_PATH, OUTPUT_PATH, FACE_ANALYSIS, FACE_SWAPPER):
    kirk_face = get_random_kirk_face(FACE_ANALYSIS)

    print("Kirkifying...")
    FACE_DETECTED = kirkify_frame(TARGET_PATH, OUTPUT_PATH, FACE_ANALYSIS, FACE_SWAPPER, kirk_face)

    if not FACE_DETECTED:
        print("No faces detected. Image unchanged.")

    print(f"Done! Output saved to {OUTPUT_PATH}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        initialize_faceanalysis_and_swapper()
        print("Initialized!")
        exit()

    if len(sys.argv) < 2:
        print("Usage: python3 kirkify.py <input_media> [output_path] [--fast] [--frame-step N] [--workers M]")
        exit(1)

    TARGET_PATH = sys.argv[1]

    if not Path(TARGET_PATH).exists():
        print("ERROR: target path not real")
        exit(1)
    
    IS_IMAGE = False
    FILE_EXT = os.path.splitext(TARGET_PATH)[1].lower()

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.flv'}

    if FILE_EXT in image_extensions:
        IS_IMAGE = True
    elif FILE_EXT in video_extensions:
        IS_IMAGE = False
    else:
        raise ValueError('Must be an image or video.')

    if len(sys.argv) > 2 and not sys.argv[2].startswith('-'):
        OUTPUT_PATH = sys.argv[2]
    else:
        OUTPUT_PATH = f"output{FILE_EXT}"

    # Performance flags
    FAST = '--fast' in sys.argv
    STEP = 1
    WORKERS = None
    for i, arg in enumerate(sys.argv):
        if arg == '--frame-step' and i + 1 < len(sys.argv):
            try:
                STEP = max(1, int(sys.argv[i+1]))
            except ValueError:
                pass
        if arg == '--workers' and i + 1 < len(sys.argv):
            try:
                WORKERS = max(1, int(sys.argv[i+1]))
            except ValueError:
                pass

    det = (320, 320) if FAST else (640, 640)
    # Explicit CPU/GPU selection for FaceAnalysis and ONNX Runtime
    USE_GPU = '--gpu' in sys.argv
    USE_INTEL = '--intel' in sys.argv
    USE_CPU = '--cpu' in sys.argv
    ctx_id = 0 if (USE_GPU or USE_INTEL) and not USE_CPU else (-1 if USE_CPU else 0)
    providers = None
    if USE_GPU and not USE_CPU:
        providers = ['CUDAExecutionProvider']
    elif USE_INTEL and not USE_GPU and not USE_CPU:
        providers = [('OpenVINOExecutionProvider', {'device_type': 'GPU'}), 'CPUExecutionProvider']
    elif USE_CPU:
        providers = ['CPUExecutionProvider']

    # Initialize models once
    # Pre-check ONNX Runtime availability and the requested hardware provider.
    ok, avail = ort_available_and_providers()
    if USE_GPU and (not ok or 'CUDAExecutionProvider' not in avail):
        print("[WARN] --gpu requested, but ONNX Runtime GPU is not available; falling back to CPU.")
        providers = ['CPUExecutionProvider']
        ctx_id = -1
    elif USE_INTEL and not USE_GPU and (not ok or 'OpenVINOExecutionProvider' not in avail):
        print("[WARN] --intel requested, but ONNX Runtime OpenVINO is not available; falling back to CPU.")
        providers = ['CPUExecutionProvider']
        ctx_id = -1

    device_label = "Intel GPU" if USE_INTEL and ctx_id == 0 else ("GPU" if ctx_id == 0 else "CPU")
    print("Initializing models..." + (" (fast mode)" if FAST else "") + f" [{device_label}]")
    with suppress_output():
        FACE_ANALYSIS, FACE_SWAPPER = initialize_faceanalysis_and_swapper(det_size=det, ctx_id=ctx_id, swapper_providers=providers)

    # Report active providers for INSwapper
    sess = getattr(FACE_SWAPPER, 'session', None) or getattr(FACE_SWAPPER, 'model', None)
    active = get_session_providers(sess) if sess is not None else []
    if active:
        print(f"INSwapper active providers: {active}")
    else:
        if providers:
            print(f"INSwapper active providers unavailable; attempted: {providers}")
    for model_name, model_providers in get_faceanalysis_providers(FACE_ANALYSIS).items():
        print(f"FaceAnalysis {model_name} active providers: {model_providers}")

    try:
        if IS_IMAGE:
            kirkify_image(TARGET_PATH, OUTPUT_PATH, FACE_ANALYSIS, FACE_SWAPPER)
        else:
            kirkify_video(TARGET_PATH, OUTPUT_PATH, FACE_ANALYSIS, FACE_SWAPPER, frame_step=STEP, workers=WORKERS)
    except KeyboardInterrupt:
        cleanup('audio.aac')
        print("Goodbye!")

if __name__ == "__main__":
    main()