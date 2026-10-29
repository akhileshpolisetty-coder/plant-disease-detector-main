"""
Plant Disease Detection — Streamlit Inference App
Run: python -m streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from PIL import Image, ImageFilter, ImageOps, ImageStat

# ── path bootstrap ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from model_transfer import get_resnet_model  # type: ignore[import]

MODELS_DIR      = BASE_DIR / "models"
PROCESSED_TRAIN = BASE_DIR / "data" / "processed" / "train"
MIN_MODEL_SCORE = 0.65
MIN_PLANT_SCORE = 0.002
MIN_IMAGE_DETAIL_STDDEV = 8.0
MIN_SHARPNESS_SCORE = 400.0
ENHANCE_SHARPNESS_SCORE = 1000.0
PLANT_IMAGENET_LABELS = {
    "head cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash",
    "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper",
    "Granny Smith", "strawberry", "orange", "lemon", "fig", "pineapple",
    "banana", "jackfruit", "custard apple", "pomegranate", "daisy", "corn",
    "acorn", "buckeye",
}

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",  # wide layout for the dashboard split
)

# ── CSS (Minimal Sage & Dark Slate Theme) ──────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }

.stApp {
    background: #090d12;
    color: #c9d1d9;
}

/* Limit content width on ultra-wide screens for better look */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ── top bar ── */
.topbar {
    background: #11161d;
    border: 1px solid #1c222c;
    padding: 1.25rem 2rem;
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin-bottom: 1.5rem;
    border-radius: 12px;
}
.topbar-icon { font-size: 1.8rem; line-height: 1; }
.topbar-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #6e8574; /* desaturated sage green */
    margin: 0;
}
.topbar-sub {
    font-size: 0.82rem;
    color: #6e7a86;
    margin: 0;
}

/* ── section heading ── */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6e7a86;
    margin-bottom: 0.6rem;
}

/* ── card container ── */
.dashboard-card {
    background: #11161d;
    border-radius: 14px;
    border: 1px solid #1c222c;
    padding: 1.4rem 1.6rem;
    height: 100%;
}

/* ── upload zone ── */
[data-testid="stFileUploader"] > div {
    border: 1.5px dashed rgba(110, 133, 116, 0.2) !important;
    border-radius: 10px !important;
    background: #161c24 !important;
    transition: border-color 0.2s, background-color 0.2s;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: #6e8574 !important;
    background: #1a232d !important;
}

/* ── status pill ── */
.pill {
    display: inline-block;
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
}
.pill-healthy { background: rgba(110, 133, 116, 0.08); color: #6e8574; border: 1px solid rgba(110, 133, 116, 0.25); }
.pill-disease { background: rgba(223, 177, 91, 0.08); color: #dfb15b; border: 1px solid rgba(223, 177, 91, 0.25); }

/* ── prediction name ── */
.pred-plant   { font-size: 0.88rem; font-weight: 600; color: #6e7a86; margin-bottom: 0.1rem; }
.pred-disease { font-size: 1.4rem; font-weight: 700; color: #e6edf3; line-height: 1.25; margin-bottom: 0.3rem; }
.pred-conf    { font-size: 0.85rem; color: #6e7a86; margin-bottom: 1.2rem; }
.pred-conf strong { color: #6e8574; }

/* ── confidence bars ── */
.bar-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.55rem;
}
.bar-label {
    font-size: 0.78rem;
    color: #8b949e;
    width: 180px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex-shrink: 0;
}
.bar-track {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #2a3c35, #6e8574);
}
.bar-pct {
    font-size: 0.78rem;
    font-weight: 600;
    color: #6e8574;
    width: 38px;
    text-align: right;
    flex-shrink: 0;
}

/* ── divider ── */
.thin-hr { border: none; border-top: 1px solid #1c222c; margin: 1.1rem 0; }

/* ── info chip ── */
.chip {
    display: inline-block;
    background: rgba(110, 133, 116, 0.05);
    border: 1px solid rgba(110, 133, 116, 0.12);
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    font-size: 0.72rem;
    color: #84a98c;
    font-weight: 500;
    margin-right: 0.3rem;
}
.chip-model {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    color: #8b949e;
}

/* ── placeholder & image frames ── */
.placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    min-height: 280px;
    color: #6e7a86;
}
.placeholder .icon { font-size: 2.8rem; margin-bottom: 0.6rem; opacity: 0.6; }
.placeholder p { font-size: 0.9rem; margin: 0; }

.image-frame {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #1c222c;
    background: #090d12;
    margin-bottom: 0.5rem;
}

/* ── warning box ── */
.warn-box {
    background: rgba(223, 177, 91, 0.05);
    border: 1px solid rgba(223, 177, 91, 0.2);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #dfb15b;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_class_names():
    class_file = MODELS_DIR / "class_names.json"
    if not class_file.exists():
        return []
    import json
    with open(class_file) as f:
        return json.load(f)


@st.cache_resource
def load_model(num_classes: int):
    candidates = [
        MODELS_DIR / "resnet_transfer_best.pth",
        MODELS_DIR / "resnet_transfer_phase1_best.pth",
    ]
    mismatches = []

    for ckpt_path in candidates:
        if not ckpt_path.exists():
            continue
        try:
            state_dict = torch.load(
                ckpt_path, map_location="cpu", weights_only=True
            )
        except (OSError, RuntimeError, EOFError) as exc:
            return None, f"Could not load `{ckpt_path.name}`: {exc}", None

        classifier_weight = state_dict.get("fc.weight")
        if not isinstance(classifier_weight, torch.Tensor):
            return None, f"`{ckpt_path.name}` is missing the ResNet classifier weights.", None
        if classifier_weight.shape[0] != num_classes:
            mismatches.append(
                f"{ckpt_path.name} has {classifier_weight.shape[0]} classes"
            )
            continue

        model = get_resnet_model(num_classes=num_classes, freeze_base=False)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            return None, f"Checkpoint `{ckpt_path.name}` is incompatible: {exc}", None
        model.eval()
        notice = None
        if ckpt_path.name != candidates[0].name:
            notice = (
                "The final checkpoint has the wrong number of classes; using the "
                "compatible phase-1 checkpoint instead. Retrain the transfer model "
                "to produce a matching final checkpoint."
            )
        return model, None, notice

    if mismatches:
        detail = "; ".join(mismatches)
        error = (
            f"No checkpoint matches the {num_classes} configured classes ({detail}). "
            "Run `python src/run_training.py --model transfer --epochs 10 --lr 0.001`."
        )
    else:
        error = (
            "No ResNet checkpoint found. Train a model with "
            "`python src/run_training.py --model transfer --epochs 10 --lr 0.001`."
        )
    return None, error, None


@st.cache_resource
def load_plant_image_validator():
    weights = ResNet18_Weights.DEFAULT
    categories = weights.meta["categories"]
    plant_indices = [
        index for index, category in enumerate(categories)
        if category in PLANT_IMAGENET_LABELS
    ]
    if not plant_indices:
        raise RuntimeError("No plant categories are available in the image validator.")

    validator = resnet18(weights=weights)
    validator.eval()
    return validator, weights.transforms(), plant_indices


def is_plant_image(image: Image.Image, validator, preprocess, plant_indices) -> tuple[bool, float]:
    rgb_image = image.convert("RGB")
    if max(ImageStat.Stat(rgb_image).stddev) < MIN_IMAGE_DETAIL_STDDEV:
        return False, 0.0

    tensor = preprocess(rgb_image).unsqueeze(0)
    with torch.inference_mode():
        scores = F.softmax(validator(tensor), dim=1)[0]
    plant_score = scores[plant_indices].sum().item()
    return plant_score >= MIN_PLANT_SCORE, plant_score


def prepare_leaf_image(image: Image.Image) -> tuple[Image.Image, float, bool]:
    grayscale = image.convert("L").resize((256, 256), Image.Resampling.BILINEAR)
    sharpness_score = ImageStat.Stat(
        grayscale.filter(ImageFilter.FIND_EDGES)
    ).var[0]

    if sharpness_score < MIN_SHARPNESS_SCORE:
        return image, sharpness_score, False
    if sharpness_score < ENHANCE_SHARPNESS_SCORE:
        sharpened = image.filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=125, threshold=3)
        )
        return sharpened, sharpness_score, True
    return image, sharpness_score, False


INFER_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_with_attention(model, image: Image.Image, top_k: int = 5):
    t = INFER_TF(image.convert("RGB")).unsqueeze(0)
    activations = {}

    def capture_activations(module, inputs, output):
        output.retain_grad()
        activations["features"] = output

    hook = model.layer4.register_forward_hook(capture_activations)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(t)
            predicted_index = logits.argmax(dim=1).item()
            logits[0, predicted_index].backward()
        probs = F.softmax(logits.detach(), dim=1)[0]
        top = probs.topk(min(top_k, len(probs)))

        features = activations["features"]
        gradients = features.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = F.relu((weights * features).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(
            heatmap, size=(224, 224), mode="bilinear", align_corners=False
        )[0, 0]
        heatmap = heatmap / heatmap.max().clamp(min=1e-8)
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)

    grayscale = Image.frombytes(
        "L",
        (224, 224),
        heatmap.mul(255).byte().cpu().numpy().tobytes(),
    ).resize(image.size, Image.Resampling.BILINEAR)
    colors = ImageOps.colorize(
        grayscale, black="#001a33", mid="#ffd166", white="#ef233c"
    )
    alpha = grayscale.point(lambda value: int(value * 0.55))
    overlay = Image.merge("RGBA", (*colors.split(), alpha))
    overlay = Image.alpha_composite(image.convert("RGBA"), overlay)

    predictions = list(zip(top.indices.tolist(), top.values.tolist()))
    return predictions, overlay.convert("RGB")


def split_label(raw: str) -> tuple[str, str]:
    parts = raw.split("___", 1)
    plant   = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return plant, disease


# ── top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-icon">🌿</div>
  <div>
    <p class="topbar-title">Plant Disease Detector</p>
    <p class="topbar-sub">AI leaf screening · 38 PlantVillage classes · PyTorch</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Screening aid only: this model was trained on controlled PlantVillage images, "
    "not validated on field photos. It can miss diseases or give a confident "
    "wrong result, especially for poor-quality or unfamiliar images. Confirm "
    "important decisions with a qualified local plant-health expert."
)

# ── load classes & model ──────────────────────────────────────────────────────
classes = load_class_names()
model, err, model_notice = (
    load_model(num_classes=len(classes)) if classes else (None, None, None)
)

# ── main layout split ─────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2.2], gap="large")

# ── left panel: input & upload ──
with left_col:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Pipeline Info</p>', unsafe_allow_html=True)
    
    # model and session chips
    st.markdown('<div style="margin-bottom: 1.25rem;">', unsafe_allow_html=True)
    st.markdown('<span class="chip chip-model">ResNet18 (Transfer)</span>', unsafe_allow_html=True)
    if classes:
        st.markdown(f'<span class="chip">{len(classes)} classes</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="chip">{"GPU" if torch.cuda.is_available() else "CPU"}</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not classes:
        st.markdown('<div class="warn-box">⚠️ No processed data found. Run <code>python src/make_subset.py</code> first.</div>',
                    unsafe_allow_html=True)
    elif err:
        st.markdown(f'<div class="warn-box">⚠️ {err}</div>', unsafe_allow_html=True)
    elif model_notice:
        st.warning(model_notice)
    
    st.markdown('<p class="section-label">Upload leaf image</p>', unsafe_allow_html=True)
    st.caption(
        "For a better result, photograph one whole leaf in focus, in natural light, "
        "with minimal background. The plant-image check is best-effort, so unusual "
        "plant photos may still be rejected."
    )
    uploaded = st.file_uploader(
        "leaf_upload",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    
    st.markdown('<div style="margin-top: 1rem;"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── right panel: prediction results ──
with right_col:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Inference Diagnosis</p>', unsafe_allow_html=True)
    
    # Create sub-columns for visual preview and diagnosis details side-by-side
    img_sub_col, info_sub_col = st.columns([1, 1.1], gap="medium")
    
    image = None
    sharpness_score = None
    was_sharpened = False
    upload_error = None
    if uploaded:
        try:
            with Image.open(uploaded) as opened_image:
                opened_image.load()
                original_image = ImageOps.exif_transpose(opened_image).convert("RGB")
            image, sharpness_score, was_sharpened = prepare_leaf_image(original_image)
        except (Image.UnidentifiedImageError, OSError, ValueError) as exc:
            upload_error = f"Could not read this image: {exc}"
        
    with img_sub_col:
        if image is not None:
            st.markdown('<p class="section-label">Leaf Photo</p>', unsafe_allow_html=True)
            st.markdown('<div class="image-frame">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(
                "Preview with mild sharpening applied."
                if was_sharpened else "Original photo"
            )
        else:
            st.markdown("""
            <div class="placeholder" style="min-height: 250px;">
              <div class="icon">📷</div>
              <p>Leaf image preview will appear here</p>
            </div>
            """, unsafe_allow_html=True)
            
    with info_sub_col:
        if upload_error:
            st.error(upload_error)
        elif image is not None and classes and model is not None:
            if sharpness_score is not None and sharpness_score < MIN_SHARPNESS_SCORE:
                st.error(
                    "This photo is too blurry to analyze reliably, so no disease "
                    "prediction was made. Retake it with the camera focused on the "
                    "leaf, hold the camera steady, and use good lighting."
                )
            else:
                validation_error = None
                try:
                    plant_validator, plant_preprocess, plant_indices = load_plant_image_validator()
                    is_plant, plant_score = is_plant_image(
                        image, plant_validator, plant_preprocess, plant_indices
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    validation_error = exc
                    plant_score = 0.0

                if validation_error is not None:
                    st.error(f"Could not verify that this is a plant image: {validation_error}")
                elif not is_plant:
                    st.error(
                        "This image was not recognized as a plant image, so no disease "
                        "prediction was made. Upload a clear photo of a plant or leaf. "
                        f"(Plant-image score: {plant_score:.1%})"
                    )
                else:
                    if was_sharpened:
                        st.warning(
                            "This photo looked slightly soft, so mild sharpening was "
                            "applied. Sharpening cannot restore missing detail; retake "
                            "the photo if possible."
                        )
                    with st.spinner("Analysing leaf symptoms…"):
                        top_preds, attention_image = predict_with_attention(
                            model, image, top_k=5
                        )

                    top_idx, top_conf = top_preds[0]
                    plant, disease = split_label(classes[top_idx])
                    is_healthy = disease.casefold() == "healthy"
                    st.markdown("**Approximate region the model focused on**")
                    st.image(attention_image, use_container_width=True)
                    st.caption(
                        "Warm colors show areas that influenced this class prediction. "
                        "This is a coarse attention map, not a measured disease boundary."
                    )
                    if top_conf < MIN_MODEL_SCORE:
                        st.warning(
                            f"Uncertain result. The top candidate is {plant} — {disease}. "
                            "Try a clear, close-up photo of one leaf in natural light "
                            "and compare the alternatives below."
                        )
                    elif is_healthy:
                        st.success("Leaf status: likely healthy")
                        st.caption(f"Top match: {plant} — {disease}")
                    else:
                        st.warning("Leaf status: possible issue detected")
                        st.markdown(f"**Likely issue:** {disease}  \n**Plant:** {plant}")
                    st.caption(
                        f"Model score: {top_conf * 100:.1f}% · This is an image-classifier "
                        "score, not a calibrated probability or a confirmed diagnosis."
                    )

                    if (
                        top_conf >= MIN_MODEL_SCORE
                        and classes[top_idx] == "Tomato___Bacterial_spot"
                    ):
                        st.info(
                            "**Possible tomato bacterial spot.** Look for small, dark, "
                            "water-soaked spots that may develop yellow halos. Similar "
                            "symptoms can have other causes, so this photo cannot confirm "
                            "the pathogen. Avoid overhead watering, clean tools between "
                            "plants, and consult a local plant-health extension service "
                            "before treating."
                        )

                    for idx, conf in top_preds:
                        p, d = split_label(classes[idx])
                        label = f"{p} · {d}"
                        pct   = conf * 100
                        st.markdown(f"""
                      <div class="bar-row">
                        <div class="bar-label" title="{label}">{label}</div>
                        <div class="bar-track">
                          <div class="bar-fill" style="width:{pct:.1f}%"></div>
                        </div>
                        <div class="bar-pct">{pct:.0f}%</div>
                      </div>
                        """, unsafe_allow_html=True)
        elif image is not None and not classes:
            st.warning("Cannot check this image until the plant classes are configured.")
                
        elif err:
            st.markdown(f"""
            <div class="placeholder" style="min-height: 250px;">
              <div class="icon" style="color: #dfb15b;">⚠️</div>
              <p>Model loading issue must be resolved to run prediction</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="placeholder" style="min-height: 250px;">
              <div class="icon">📊</div>
              <p>Provide a leaf photo on the left to display diagnosis</p>
            </div>
            """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
