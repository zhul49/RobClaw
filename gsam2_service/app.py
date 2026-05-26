import os
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import Response
from torchvision.ops import box_convert

# Grounded-SAM-2 demo imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from grounding_dino.groundingdino.util.inference import load_model, predict

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

BOX_THRESHOLD_DEFAULT = 0.35
TEXT_THRESHOLD_DEFAULT = 0.25
MULTIMASK_OUTPUT = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# App + global models
# -----------------------
app = FastAPI()

sam2_model = None                    # kept for SAM2AutomaticMaskGenerator construction
sam2_predictor: SAM2ImagePredictor | None = None
sam2_auto_generator: SAM2AutomaticMaskGenerator | None = None
grounding_model = None


def _ensure_models_loaded() -> None:
    global sam2_model, sam2_predictor, sam2_auto_generator, grounding_model
    if sam2_predictor is not None and grounding_model is not None:
        return

    # Locate Grounded-SAM-2 repo even if uvicorn launched elsewhere
    repo_root = Path(__file__).resolve().parent.parent / "Grounded-SAM-2"
    candidates = [
        Path.cwd(),
        Path.home() / "Grounded-SAM-2",
        repo_root,
    ]
    repo = None
    for c in candidates:
        if (c / "sam2").exists() and (c / "grounding_dino").exists():
            repo = c
            break
    if repo is None:
        raise RuntimeError("Could not locate Grounded-SAM-2 repo directory")

    # Work in repo so relative paths (./checkpoints/...) match the demo
    os.chdir(str(repo))

    # Inference-only
    torch.set_grad_enabled(False)

    # Build SAM2 predictor (text-prompted) AND automatic mask generator (paper-style "all masks")
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    # Defaults tuned to match SAM-everything: 32x32 point grid, IoU thr 0.8.
    # min_mask_region_area=100 drops tiny noise blobs that the proposer would
    # later cluster into nothing useful.
    sam2_auto_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
        output_mode="binary_mask",
    )

    # Build GroundingDINO model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )

    # TF32 is fine (Ampere+). This is NOT BF16 autocast.
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if getattr(props, "major", 0) >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


@app.on_event("startup")
def startup_event():
    _ensure_models_loaded()


def _decode_image_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image")
    return bgr


def _mask_to_png_bytes(mask_u8: np.ndarray) -> bytes:
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    ok, png = cv2.imencode(".png", mask_u8)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return png.tobytes()


def _autocast_ctx():
    """
    GroundingDINO ms_deform_attn CUDA op often does NOT support BF16.
    Force FP16 autocast on CUDA to avoid BF16 path.
    """
    if DEVICE == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


@app.post("/predict_mask")
async def predict_mask(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    box_threshold: float = Form(BOX_THRESHOLD_DEFAULT),
    text_threshold: float = Form(TEXT_THRESHOLD_DEFAULT),
):
    """
    Returns a single-channel PNG mask (0/255) for the best detection of `prompt`.
    NOTE: prompt must be lowercased and end with a dot, e.g. "phone." "egg."
    """
    _ensure_models_loaded()
    assert sam2_predictor is not None and grounding_model is not None

    data = await image.read()
    bgr = _decode_image_to_bgr(data)

    # SAM2 expects RGB uint8 HWC
    image_source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # GroundingDINO expects a preprocessed tensor like load_image() returns.
    import grounding_dino.groundingdino.datasets.transforms as T
    from PIL import Image as PILImage

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    pil = PILImage.fromarray(image_source)
    image_tensor, _ = transform(pil, None)  # torch.Tensor CHW float
    image_tensor = image_tensor.to(DEVICE)  # IMPORTANT: move to GPU if cuda

    # Text must be lowercased + end with dot
    text = prompt.strip()
    if text and (text[-1] != "."):
        text = text + "."
    text = text.lower()

    # Set image for SAM2
    sam2_predictor.set_image(image_source)

    # Run GroundingDINO with a safe dtype context (FP16 on CUDA, not BF16)
    with torch.inference_mode(), _autocast_ctx():
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image_tensor,
            caption=text,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            device=DEVICE,
        )

    # No detections -> empty mask
    h, w, _ = image_source.shape
    if boxes is None or len(boxes) == 0:
        empty = np.zeros((h, w), dtype=np.uint8)
        return Response(content=_mask_to_png_bytes(empty), media_type="image/png")

    # Convert boxes to xyxy pixels
    boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").detach().cpu().numpy()

    # Pick best box by confidence
    conf_np = confidences.detach().cpu().numpy()
    best_i = int(np.argmax(conf_np))
    best_box = input_boxes[best_i : best_i + 1]  # shape (1,4)

    # Run SAM2 (FP16 autocast is typically fine; if it ever breaks, we can disable it just here)
    with torch.inference_mode(), _autocast_ctx():
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=best_box,
            multimask_output=MULTIMASK_OUTPUT,
        )

    # masks can be (N,1,H,W) or (N,H,W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # We requested a single box; masks is (1,H,W)
    mask = masks[0].astype(np.uint8) * 255
    return Response(content=_mask_to_png_bytes(mask), media_type="image/png")



@app.post("/predict_all_masks")
async def predict_all_masks(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    box_threshold: float = Form(BOX_THRESHOLD_DEFAULT),
    text_threshold: float = Form(TEXT_THRESHOLD_DEFAULT),
):
    """
    Returns a single-channel PNG where pixel value n = nth detected instance (1-indexed).
    0 = background. Up to 254 instances supported.
    Also returns metadata as JSON in response headers: 'X-Detections' (count),
    'X-Confidences' (comma-separated), 'X-Boxes' (comma-separated xyxy quads).
    """
    _ensure_models_loaded()
    assert sam2_predictor is not None and grounding_model is not None

    data = await image.read()
    bgr = _decode_image_to_bgr(data)
    image_source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    import grounding_dino.groundingdino.datasets.transforms as T
    from PIL import Image as PILImage

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pil = PILImage.fromarray(image_source)
    image_tensor, _ = transform(pil, None)
    image_tensor = image_tensor.to(DEVICE)

    text = prompt.strip().lower()
    if text and text[-1] != ".":
        text = text + "."

    sam2_predictor.set_image(image_source)

    with torch.inference_mode(), _autocast_ctx():
        boxes, confidences, labels = predict(
            model=grounding_model,
            image=image_tensor,
            caption=text,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
            device=DEVICE,
        )

    h, w, _ = image_source.shape
    multi_mask = np.zeros((h, w), dtype=np.uint8)
    headers = {"X-Detections": "0", "X-Confidences": "", "X-Boxes": "", "X-Labels": ""}

    if boxes is None or len(boxes) == 0:
        return Response(content=_mask_to_png_bytes(multi_mask),
                        media_type="image/png", headers=headers)

    boxes_px = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    input_boxes = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").detach().cpu().numpy()
    conf_np = confidences.detach().cpu().numpy()

    # convert labels to plain python list for ordering
    if hasattr(labels, "cpu"):
        labels_list = labels.cpu().tolist()
    else:
        labels_list = list(labels)
    # sort by confidence descending so highest-conf gets lowest label
    order = np.argsort(-conf_np)
    input_boxes = input_boxes[order]
    conf_np = conf_np[order]
    labels_list = [labels_list[i] for i in order]

    # cap at 254 to fit in uint8 (255 reserved for clarity)
    n_max = min(254, len(input_boxes))
    input_boxes = input_boxes[:n_max]
    conf_np = conf_np[:n_max]
    labels_list = labels_list[:n_max]

    # SAM2 batched predict for all boxes at once
    with torch.inference_mode(), _autocast_ctx():
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks.squeeze(1)
    # masks shape: (N, H, W)

    # paint each detection with its label, lowest-conf last so highest-conf wins overlaps
    for i in range(len(masks) - 1, -1, -1):
        binary = masks[i].astype(bool)
        multi_mask[binary] = i + 1  # 1-indexed

    # build headers
    headers["X-Detections"] = str(len(masks))
    headers["X-Confidences"] = ",".join(f"{c:.3f}" for c in conf_np)
    headers["X-Boxes"] = ";".join(",".join(f"{v:.1f}" for v in box) for box in input_boxes)
    # use | as separator since labels can contain commas/semicolons
    headers["X-Labels"] = "|".join(str(l) for l in labels_list)

    return Response(content=_mask_to_png_bytes(multi_mask),
                    media_type="image/png", headers=headers)


@app.post("/predict_automatic_masks")
async def predict_automatic_masks(
    image: UploadFile = File(...),
    points_per_side: int = Form(32),
    pred_iou_thresh: float = Form(0.8),
    stability_score_thresh: float = Form(0.95),
    min_mask_region_area: int = Form(100),
):
    """
    SAM-everything: no text prompt, no boxes. Tiles a point grid across the
    image and returns every mask SAM2 produces. Used by the paper-faithful
    keypoint proposer (ReKep paper Section A.5).

    Returns single-channel PNG, pixel value n = nth mask (1-indexed), 0 = bg.
    Sorted largest-first so smaller masks win overlap (paint largest last).
    Up to 254 masks. Headers: X-Detections (count), X-Areas (comma-separated).
    """
    _ensure_models_loaded()
    assert sam2_auto_generator is not None

    # Allow per-request overrides without rebuilding the generator if the
    # caller asks for the defaults. Otherwise build a temp generator —
    # rebuilds are cheap because the underlying SAM2 weights are already on GPU.
    gen = sam2_auto_generator
    use_defaults = (
        points_per_side == 32 and pred_iou_thresh == 0.8
        and stability_score_thresh == 0.95 and min_mask_region_area == 100
    )
    if not use_defaults:
        gen = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=int(points_per_side),
            pred_iou_thresh=float(pred_iou_thresh),
            stability_score_thresh=float(stability_score_thresh),
            min_mask_region_area=int(min_mask_region_area),
            output_mode="binary_mask",
        )

    data = await image.read()
    bgr = _decode_image_to_bgr(data)
    image_source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with torch.inference_mode(), _autocast_ctx():
        results = gen.generate(image_source)

    h, w, _ = image_source.shape
    multi_mask = np.zeros((h, w), dtype=np.uint8)
    headers = {"X-Detections": "0", "X-Areas": ""}
    if not results:
        return Response(content=_mask_to_png_bytes(multi_mask),
                        media_type="image/png", headers=headers)

    # Sort smallest-first so largest paints LAST and wins overlap. (We want
    # smaller foreground masks to survive when nested inside larger ones.)
    results = sorted(results, key=lambda r: int(r["area"]))
    n_max = min(254, len(results))
    results = results[:n_max]
    for i, r in enumerate(results):
        seg = r["segmentation"].astype(bool)
        multi_mask[seg] = i + 1

    headers["X-Detections"] = str(len(results))
    headers["X-Areas"] = ",".join(str(int(r["area"])) for r in results)
    return Response(content=_mask_to_png_bytes(multi_mask),
                    media_type="image/png", headers=headers)


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}
