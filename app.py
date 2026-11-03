# pyright: reportMissingImports=false
import asyncio
import base64
import io
import inspect
import os
import secrets
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from PIL import Image, ImageOps
import requests

load_dotenv(Path(__file__).with_name(".env"))

app = FastAPI(title="VASTRA Virtual Try-On API")

# Gradio Client prints Unicode status symbols while connecting to Spaces. On
# Windows, the default console encoding may not be able to write those symbols.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _create_hf_client(space: str, hf_token: str | None) -> Client:
    """Support gradio_client releases that call the token argument hf_token."""
    parameters = inspect.signature(Client).parameters
    if "token" in parameters:
        return Client(space, token=hf_token)
    if "hf_token" in parameters:
        return Client(space, hf_token=hf_token)
    return Client(space)


def _write_upload(data: bytes, suffix: str) -> str:
    file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.write(data)
        return file.name
    finally:
        file.close()


_cpu_inpaint_pipeline = None


def _get_cpu_inpaint_pipeline():
    global _cpu_inpaint_pipeline
    if _cpu_inpaint_pipeline is not None:
        return _cpu_inpaint_pipeline

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    model_id = os.environ.get(
        "LOCAL_CPU_INPAINT_MODEL",
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
    )
    _cpu_inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        safety_checker=None,
    )
    _cpu_inpaint_pipeline = _cpu_inpaint_pipeline.to("cpu")
    if hasattr(_cpu_inpaint_pipeline, "enable_attention_slicing"):
        _cpu_inpaint_pipeline.enable_attention_slicing("max")
    # Diffusers versions expose VAE tiling on either the pipeline or the VAE,
    # and some supported pipeline versions expose neither method.
    if hasattr(_cpu_inpaint_pipeline, "enable_vae_tiling"):
        _cpu_inpaint_pipeline.enable_vae_tiling()
    elif hasattr(_cpu_inpaint_pipeline.vae, "enable_tiling"):
        _cpu_inpaint_pipeline.vae.enable_tiling()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return _cpu_inpaint_pipeline


def _dominant_garment_color(image: Image.Image) -> str:
    sample = image.convert("RGB").resize((64, 64))
    pixels = list(sample.getdata())
    if not pixels:
        return "suitable traditional saree colors"

    def is_near_white(pixel):
        return pixel[0] > 238 and pixel[1] > 238 and pixel[2] > 238

    useful = [pixel for pixel in pixels if not is_near_white(pixel)][:400]
    if not useful:
        useful = pixels
    red = sum(pixel[0] for pixel in useful) // len(useful)
    green = sum(pixel[1] for pixel in useful) // len(useful)
    blue = sum(pixel[2] for pixel in useful) // len(useful)

    palette = [
        ("black", (0, 0, 0)),
        ("white", (255, 255, 255)),
        ("red", (220, 35, 35)),
        ("maroon", (128, 0, 32)),
        ("pink", (230, 100, 150)),
        ("orange", (240, 130, 40)),
        ("yellow", (230, 200, 40)),
        ("green", (35, 130, 70)),
        ("blue", (40, 90, 180)),
        ("purple", (120, 60, 160)),
        ("gold", (205, 170, 55)),
    ]
    closest = min(
        palette,
        key=lambda item: sum((item[1][index] - value) ** 2 for index, value in enumerate((red, green, blue))),
    )
    return closest[0]


def _run_cpu_inpaint(person_path: str, cloth_path: str) -> tuple[str, str]:
    from PIL import ImageDraw, ImageFilter

    person = Image.open(person_path).convert("RGB")
    garment = Image.open(cloth_path).convert("RGB")
    width, height = 512, 768
    person = ImageOps.fit(person, (width, height), method=Image.Resampling.LANCZOS)
    garment = ImageOps.fit(garment, (width, height), method=Image.Resampling.LANCZOS)

    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(
        [
            (int(width * 0.27), int(height * 0.26)),
            (int(width * 0.73), int(height * 0.26)),
            (int(width * 0.94), int(height * 0.41)),
            (int(width * 0.90), int(height * 0.97)),
            (int(width * 0.10), int(height * 0.97)),
            (int(width * 0.06), int(height * 0.41)),
        ],
        fill=255,
    )
    # Feather the replacement boundary to reduce a visible straight seam.
    mask = mask.filter(ImageFilter.GaussianBlur(radius=7))

    color_name = _dominant_garment_color(garment)
    prompt = (
        "A realistic full-length traditional Indian silk saree with a matching fitted blouse, "
        f"main {color_name} fabric, gold zari border, natural fabric folds and "
        "realistic studio lighting. Preserve the person's face, hairstyle, skin tone, "
        "pose, hands, feet, and background exactly. Replace the existing shirt and trousers "
        "with one continuous saree outfit, from the shoulders to the ankles."
    )
    negative_prompt = (
        "pants, trousers, shorts, t-shirt, shirt, unchanged top, horizontal clothing seam, "
        "cut-off garment, flat overlay, cutout, duplicate face, changed face, changed body shape, "
        "changed pose, changed background, extra limbs, deformed hands, blur"
    )

    pipeline = _get_cpu_inpaint_pipeline()
    generated = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=person,
        mask_image=mask,
        strength=0.9,
        guidance_scale=7.0,
        num_inference_steps=int(os.environ.get("LOCAL_CPU_INFERENCE_STEPS", "20")),
    ).images[0].convert("RGB")

    head_mask = Image.new("L", (width, height), 0)
    head_mask_draw = ImageDraw.Draw(head_mask)
    head_mask_draw.rectangle((0, 0, width, int(height * 0.28)), fill=255)
    final_image = Image.composite(person, generated, head_mask)

    output = io.BytesIO()
    final_image.save(output, format="PNG", optimize=False)
    return base64.b64encode(output.getvalue()).decode("ascii"), "image/png"


def _run_local_face_swap(source_bytes: bytes, target_bytes: bytes) -> tuple[str, str]:
    """Blend the customer's detected face into the selected saree model photo."""
    import cv2
    import numpy as np

    source = cv2.imdecode(np.frombuffer(source_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    target = cv2.imdecode(np.frombuffer(target_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source is None or target is None:
        raise ValueError("Could not read the customer or selected saree image")

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError("OpenCV face detector could not be loaded")
    eye_cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
    eye_detector = cv2.CascadeClassifier(eye_cascade_path)

    def detect_largest_face(image: np.ndarray, label: str):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        min_face_size = max(28, int(min(image.shape[:2]) * 0.035))
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size),
        )
        if len(faces) == 0:
            raise ValueError(
                f"Could not detect a clear front-facing face in the {label} image. "
                "Use a well-lit photo with the face looking toward the camera."
            )
        return max(faces, key=lambda box: int(box[2]) * int(box[3]))

    def eye_line_angle(image: np.ndarray, face_box):
        """Return the face roll angle from the best horizontal eye pair, if found."""
        x, y, face_width, face_height = map(int, face_box)
        upper_face = cv2.cvtColor(
            image[y : y + int(face_height * 0.68), x : x + face_width],
            cv2.COLOR_BGR2GRAY,
        )
        eyes = eye_detector.detectMultiScale(
            upper_face,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(max(8, face_width // 9), max(8, face_height // 12)),
        )
        centers = [
            (int(ex + ew / 2), int(ey + eh / 2))
            for ex, ey, ew, eh in eyes
            if ey + eh / 2 < face_height * 0.62
        ]
        best_pair = None
        best_score = -1.0
        for index, first in enumerate(centers):
            for second in centers[index + 1 :]:
                left_eye, right_eye = sorted((first, second), key=lambda point: point[0])
                horizontal_gap = right_eye[0] - left_eye[0]
                vertical_gap = abs(right_eye[1] - left_eye[1])
                if horizontal_gap < face_width * 0.18 or vertical_gap > face_height * 0.22:
                    continue
                score = horizontal_gap - vertical_gap * 2
                if score > best_score:
                    best_score = score
                    best_pair = (left_eye, right_eye)
        if not best_pair:
            return None
        (left_x, left_y), (right_x, right_y) = best_pair
        return float(np.degrees(np.arctan2(right_y - left_y, right_x - left_x)))

    def expanded_crop(image: np.ndarray, face_box):
        x, y, face_width, face_height = map(int, face_box)
        center_x = x + face_width // 2
        center_y = y + face_height // 2
        crop_width = int(face_width * 1.65)
        crop_height = int(face_height * 1.7)
        left = max(0, center_x - crop_width // 2)
        top = max(0, center_y - crop_height // 2)
        right = min(image.shape[1], left + crop_width)
        bottom = min(image.shape[0], top + crop_height)
        left = max(0, right - crop_width)
        top = max(0, bottom - crop_height)
        return image[top:bottom, left:right], (left, top, right, bottom)

    source_face = detect_largest_face(source, "customer")
    target_face = detect_largest_face(target, "selected saree")
    source_crop, _ = expanded_crop(source, source_face)
    target_crop, (left, top, right, bottom) = expanded_crop(target, target_face)
    crop_height, crop_width = target_crop.shape[:2]
    if crop_width < 32 or crop_height < 32:
        raise ValueError("The detected face is too small to blend reliably")

    source_aligned = cv2.resize(source_crop, (crop_width, crop_height), interpolation=cv2.INTER_CUBIC)
    source_angle = eye_line_angle(source, source_face)
    target_angle = eye_line_angle(target, target_face)
    if source_angle is not None and target_angle is not None:
        rotation = source_angle - target_angle
        if abs(rotation) <= 35:
            rotation_matrix = cv2.getRotationMatrix2D(
                (crop_width / 2, crop_height / 2),
                rotation,
                1.0,
            )
            source_aligned = cv2.warpAffine(
                source_aligned,
                rotation_matrix,
                (crop_width, crop_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT_101,
            )

    face_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    cv2.ellipse(
        face_mask,
        (crop_width // 2, crop_height // 2),
        (max(1, int(crop_width * 0.26)), max(1, int(crop_height * 0.27))),
        0,
        0,
        360,
        255,
        -1,
    )

    # Match face brightness/contrast to the catalog portrait while retaining
    # the customer's skin hue and facial details. This reduces the bright halo
    # that can appear when the two photos have different lighting.
    source_lab = cv2.cvtColor(source_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    face_pixels = face_mask > 128
    source_l = source_lab[:, :, 0][face_pixels]
    target_l = target_lab[:, :, 0][face_pixels]
    if source_l.size and target_l.size:
        source_mean, source_std = float(source_l.mean()), float(source_l.std())
        target_mean, target_std = float(target_l.mean()), float(target_l.std())
        gain = float(np.clip(target_std / max(source_std, 1.0), 0.7, 1.3))
        matched_l = (source_lab[:, :, 0] - source_mean) * gain + target_mean
        source_lab[:, :, 0] = np.clip(
            matched_l,
            0,
            255,
        )
        source_aligned = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Feather the inner-face oval so the replacement doesn't leave a bright rim.
    face_mask = cv2.GaussianBlur(
        face_mask,
        (0, 0),
        sigmaX=max(1.0, crop_width * 0.03),
        sigmaY=max(1.0, crop_height * 0.03),
    )

    alpha = (face_mask.astype(np.float32) / 255.0)[:, :, None]
    blended_crop = (
        source_aligned.astype(np.float32) * alpha
        + target_crop.astype(np.float32) * (1.0 - alpha)
    ).clip(0, 255).astype(np.uint8)

    target[top:bottom, left:right] = blended_crop
    success, encoded = cv2.imencode(".jpg", target, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError("Could not encode the face-swapped result image")
    return base64.b64encode(encoded.tobytes()).decode("ascii"), "image/jpeg"


def _run_google_tryon(person_path: str, cloth_path: str) -> tuple[str, str]:
    """Call Vertex AI's Virtual Try-On publisher model using ADC credentials."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "Google Cloud is not configured. Set GOOGLE_CLOUD_PROJECT and run "
            "'gcloud auth application-default login'."
        )

    # Import lazily so Hugging Face mode does not require Google Cloud packages.
    import google.auth
    from google.auth.transport.requests import Request as GoogleAuthRequest

    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-southeast1")
    model = os.environ.get("GOOGLE_VTO_MODEL", "virtual-try-on-001")
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(GoogleAuthRequest())

    def image_part(path: str) -> tuple[str, str]:
        with Image.open(path) as image:
            image_format = "PNG" if "A" in image.getbands() else "JPEG"
            if image_format == "JPEG":
                image = image.convert("RGB")
            with tempfile.SpooledTemporaryFile() as output:
                image.save(output, format=image_format, quality=95)
                output.seek(0)
                data = output.read()
        if len(data) > 7 * 1024 * 1024:
            raise RuntimeError("Google Virtual Try-On accepts images up to 7 MB each.")
        return base64.b64encode(data).decode("ascii"), f"image/{image_format.lower()}"

    person_data, person_mime = image_part(person_path)
    cloth_data, cloth_mime = image_part(cloth_path)
    endpoint = (
        f"https://{quote(location, safe='')}-aiplatform.googleapis.com/v1/"
        f"projects/{quote(project, safe='')}/locations/{quote(location, safe='')}/"
        f"publishers/google/models/{quote(model, safe='')}:predict"
    )
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {credentials.token}",
            "Content-Type": "application/json",
        },
        json={
            "instances": [{
                "prompt": (
                    "Create a realistic virtual try-on. Preserve the supplied garment's "
                    "colors, pattern, border, and design while dressing the person naturally."
                ),
                "personImage": {"image": {
                    "mimeType": person_mime,
                    "bytesBase64Encoded": person_data,
                }},
                "productImages": [{"image": {
                    "mimeType": cloth_mime,
                    "bytesBase64Encoded": cloth_data,
                }}],
            }],
            "parameters": {"sampleCount": 1},
        },
        timeout=(20, 300),
    )
    if not response.ok:
        # Keep Google's useful status/body for API troubleshooting.
        raise RuntimeError(
            f"Google Virtual Try-On API returned HTTP {response.status_code}: "
            f"{response.text[:1500]}"
        )

    predictions = response.json().get("predictions") or []
    if not predictions or not predictions[0].get("bytesBase64Encoded"):
        raise RuntimeError("Google Virtual Try-On returned no generated image.")
    prediction = predictions[0]
    return prediction["bytesBase64Encoded"], prediction.get("mimeType", "image/png")


def _run_hf_tryon(person_path: str, cloth_path: str) -> tuple[str, str]:
    hf_token = os.environ.get("HF_TOKEN") or None
    errors = []
    generated_path = None
    seed = secrets.randbelow(2_147_483_646) + 1

    # Sarees need a full-body transfer. IDM-VTON is an upper-body try-on model;
    # its successful responses can leave the person's trousers in place. Use
    # Leffa's dress/full-body route first and do not silently return a partial
    # result if it is unavailable.
    try:
        leffa = _create_hf_client("franciszzj/Leffa", hf_token)
        result = leffa.predict(
            src_image_path=handle_file(person_path),
            ref_image_path=handle_file(cloth_path),
            # Leffa API expects string literals 'True'/'False', NOT Python booleans
            ref_acceleration="False",
            # Higher steps → better garment-body blending quality
            step=40,
            # Higher scale → model follows the source-person's body shape
            # and face more closely, reducing the "fashion-model slim" effect.
            scale=3.5,
            seed=seed,
            # Sarees cover the torso and legs. The DressCode checkpoint is the
            # full-dress model; VITON-HD is trained for upper-body garments.
            vt_model_type="dress_code",
            vt_garment_type="dresses",
            # repaint='True' preserves the person's original body silhouette
            # instead of replacing it with the model's idealised body shape.
            vt_repaint="True",
            api_name="/leffa_predict_vt",
        )
        generated_path = result[0] if isinstance(result, (list, tuple)) else result
    except Exception as error:
        errors.append(f"Leffa full-body: {error}")

    if not generated_path or not os.path.isfile(generated_path):
        # Keep the old model available only as an explicit fallback for users
        # who prefer an image over an error, knowing that it may preserve pants.
        allow_partial = os.environ.get("TRYON_ALLOW_PARTIAL_FALLBACK", "false").lower() in {
            "1", "true", "yes"
        }
        if allow_partial:
            try:
                idm = _create_hf_client("yisol/IDM-VTON", hf_token)
                result = idm.predict(
                    dict={"background": handle_file(person_path), "layers": [], "composite": None},
                    garm_img=handle_file(cloth_path),
                    garment_des=(
                        "Traditional silk saree. Preserve the reference colors, motifs and border. "
                        "Keep the person's face, pose and background unchanged."
                    ),
                    is_checked=True,
                    is_checked_crop=False,
                    denoise_steps=30,
                    seed=seed,
                    api_name="/tryon",
                )
                generated_path = result[0] if isinstance(result, (list, tuple)) else result
            except Exception as error:
                errors.append(f"IDM-VTON partial fallback: {error}")

    if not generated_path or not os.path.isfile(generated_path):
        raise RuntimeError("Full-body saree generation failed. " + " | ".join(errors))

    with Image.open(generated_path) as image:
        mime_type = Image.MIME.get(image.format or "JPEG", "image/jpeg")
    with open(generated_path, "rb") as output:
        encoded = base64.b64encode(output.read()).decode("ascii")
    return encoded, mime_type


def _run_tryon(person_path: str, cloth_path: str) -> tuple[str, str]:
    provider = os.environ.get("TRYON_PROVIDER", "huggingface").strip().lower()
    if provider == "google":
        return _run_google_tryon(person_path, cloth_path)
    if provider in {"huggingface", "hf"}:
        return _run_hf_tryon(person_path, cloth_path)
    raise RuntimeError("TRYON_PROVIDER must be 'google' or 'huggingface'.")


@app.get("/")
def home():
    provider = os.environ.get("TRYON_PROVIDER", "huggingface").strip().lower()
    return {
        "engine": "VASTRA Virtual Try-On API",
        "status": "ok",
        "provider": "Google Cloud Vertex AI" if provider == "google" else "Hugging Face Spaces",
        "google_cloud_project_configured": bool(os.environ.get("GOOGLE_CLOUD_PROJECT")),
        "google_cloud_location": os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-southeast1"),
        "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
    }


@app.post("/local-face-swap")
async def local_face_swap(request: Request):
    try:
        data = await request.json()
        source_b64 = data.get("personImage") or data.get("sourceImage")
        target_b64 = data.get("targetImage") or data.get("baseImage")
        if not source_b64 or not target_b64:
            raise HTTPException(status_code=400, detail="personImage and targetImage are required")
        if "," in source_b64:
            source_b64 = source_b64.split(",", 1)[1]
        if "," in target_b64:
            target_b64 = target_b64.split(",", 1)[1]
        source_bytes = base64.b64decode(source_b64, validate=True)
        target_bytes = base64.b64decode(target_b64, validate=True)
        image_b64, mime_type = await asyncio.to_thread(
            _run_local_face_swap,
            source_bytes,
            target_bytes,
        )
        result_image = f"data:{mime_type};base64,{image_b64}"
        return {
            "status": "success",
            "result_image": result_image,
            "resultImage": result_image,
            "image": result_image,
            "provider": "local_face_swap",
        }
    except HTTPException:
        raise
    except Exception as error:
        print(f"Local face swap error: {error}")
        raise HTTPException(status_code=502, detail=f"Local face swap failed: {error}") from error


@app.post("/local-cpu-tryon")
async def local_cpu_tryon(
    request: Request,
    person: UploadFile | None = File(default=None),
    cloth: UploadFile | None = File(default=None),
):
    person_path = None
    cloth_path = None
    try:
        if "application/json" in request.headers.get("content-type", ""):
            data = await request.json()
            person_b64 = data.get("person_image") or data.get("personImage")
            cloth_b64 = data.get("cloth_image") or data.get("garmentImage") or data.get("garment_image")
            if not person_b64 or not cloth_b64:
                raise HTTPException(status_code=400, detail="person and cloth images required")
            if "," in person_b64:
                person_b64 = person_b64.split(",", 1)[1]
            if "," in cloth_b64:
                cloth_b64 = cloth_b64.split(",", 1)[1]
            person_bytes = base64.b64decode(person_b64, validate=True)
            cloth_bytes = base64.b64decode(cloth_b64, validate=True)
        else:
            if person is None or cloth is None:
                raise HTTPException(status_code=400, detail="person and cloth files required")
            person_bytes = await person.read()
            cloth_bytes = await cloth.read()

        person_path = _write_upload(person_bytes, "_person")
        cloth_path = _write_upload(cloth_bytes, "_cloth")
        image_b64, mime_type = await asyncio.to_thread(
            _run_cpu_inpaint,
            person_path,
            cloth_path,
        )
        return {
            "status": "success",
            "result_image": f"data:{mime_type};base64,{image_b64}",
            "resultImage": f"data:{mime_type};base64,{image_b64}",
            "image": f"data:{mime_type};base64,{image_b64}",
            "provider": "local_cpu_diffusion",
        }
    except HTTPException:
        raise
    except Exception as error:
        print(f"Local CPU try-on error: {error}")
        raise HTTPException(status_code=502, detail=f"Local CPU generation failed: {error}") from error
    finally:
        for path in (person_path, cloth_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.post("/tryon")
async def tryon(
    request: Request,
    person: UploadFile | None = File(default=None),
    cloth: UploadFile | None = File(default=None),
):
    person_path = None
    cloth_path = None
    try:
        if "application/json" in request.headers.get("content-type", ""):
            data = await request.json()
            person_b64 = data.get("person_image") or data.get("personImage")
            cloth_b64 = data.get("cloth_image") or data.get("garmentImage") or data.get("garment_image")
            if not person_b64 or not cloth_b64:
                raise HTTPException(status_code=400, detail="person and cloth images required")
            if "," in person_b64:
                person_b64 = person_b64.split(",", 1)[1]
            if "," in cloth_b64:
                cloth_b64 = cloth_b64.split(",", 1)[1]
            person_bytes = base64.b64decode(person_b64, validate=True)
            cloth_bytes = base64.b64decode(cloth_b64, validate=True)
        else:
            if person is None or cloth is None:
                raise HTTPException(status_code=400, detail="person and cloth files required")
            person_bytes = await person.read()
            cloth_bytes = await cloth.read()

        person_path = _write_upload(person_bytes, "_person")
        cloth_path = _write_upload(cloth_bytes, "_cloth")
        image_b64, mime_type = await asyncio.to_thread(
            _run_tryon, person_path, cloth_path
        )
        result_image = f"data:{mime_type};base64,{image_b64}"
        provider = os.environ.get("TRYON_PROVIDER", "huggingface").strip().lower()
        return {
            "status": "success",
            "result_image": result_image,
            "resultImage": result_image,
            "image": result_image,
            "request_id": str(uuid.uuid4()),
            "provider": "google-cloud" if provider == "google" else "huggingface",
        }
    except HTTPException:
        raise
    except Exception as error:
        print(f"Virtual try-on error: {error}")
        raise HTTPException(status_code=502, detail=f"Try-on generation failed: {error}") from error
    finally:
        for path in (person_path, cloth_path):
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
