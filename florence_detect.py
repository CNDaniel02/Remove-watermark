import os, json, math
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers import modeling_utils as _modeling_utils

MODEL_ID = os.environ.get("FLORENCE_MODEL", "multimodalart/Florence-2-large-no-flash-attn")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Compatibility shim: some custom Florence2 modules miss _supports_sdpa
if not hasattr(PreTrainedModel, "_supports_sdpa"):
    PreTrainedModel._supports_sdpa = False
if not hasattr(torch.nn.Module, "_supports_sdpa"):
    torch.nn.Module._supports_sdpa = False

# Force-disable SDPA path to avoid custom model attribute errors
def _no_sdpa(self, is_init_check: bool = False):
    return False

PreTrainedModel._sdpa_can_dispatch = _no_sdpa
_modeling_utils.PreTrainedModel._sdpa_can_dispatch = _no_sdpa

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
).to(DEVICE)
# Some remote Florence2 language backends don't inherit GenerationMixin in newer transformers
if hasattr(model, "language_model") and not isinstance(model.language_model, GenerationMixin):
    lm_cls = type(model.language_model)
    mixed_cls = type(f"{lm_cls.__name__}WithGeneration", (lm_cls, GenerationMixin), {})
    model.language_model.__class__ = mixed_cls
if hasattr(model, "language_model") and hasattr(model, "generation_config"):
    if getattr(model.language_model, "generation_config", None) is None:
        model.language_model.generation_config = model.generation_config

# Patch prepare_inputs_for_generation to handle None past_key_values in some Florence2 builds
if hasattr(model, "language_model") and hasattr(model.language_model, "prepare_inputs_for_generation"):
    _orig_prepare = model.language_model.prepare_inputs_for_generation

    def _patched_prepare_inputs_for_generation(*args, **kwargs):
        past = kwargs.get("past_key_values", None)
        if past is None and len(args) >= 2:
            past = args[1]
        if past is None:
            payload = {}
            if len(args) >= 1:
                payload["input_ids"] = args[0]
            payload["past_key_values"] = None
            payload.update(kwargs)
            return payload
        return _orig_prepare(*args, **kwargs)

    model.language_model.prepare_inputs_for_generation = _patched_prepare_inputs_for_generation
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Ensure generation config has required special tokens
_tok = getattr(processor, "tokenizer", None)
_bos = getattr(_tok, "bos_token_id", None) if _tok is not None else None
if _bos is None and _tok is not None:
    _bos = getattr(_tok, "cls_token_id", None) or getattr(_tok, "eos_token_id", None)
if _bos is not None:
    if getattr(model.generation_config, "bos_token_id", None) is None:
        model.generation_config.bos_token_id = _bos
    if getattr(model.generation_config, "decoder_start_token_id", None) is None:
        model.generation_config.decoder_start_token_id = _bos
    if hasattr(model, "language_model") and getattr(model.language_model, "generation_config", None) is not None:
        if getattr(model.language_model.generation_config, "bos_token_id", None) is None:
            model.language_model.generation_config.bos_token_id = _bos
        if getattr(model.language_model.generation_config, "decoder_start_token_id", None) is None:
            model.language_model.generation_config.decoder_start_token_id = _bos

def run_task(image: Image.Image, task_prompt: str, text_input: str | None = None):
    prompt = task_prompt if text_input is None else (task_prompt + text_input)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)

    generated = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=False,
    )
    generated_text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )
    return parsed

def clamp_bbox(b, W, H):
    x1, y1, x2, y2 = b
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]

def quad_to_bbox(q):
    # q: [x1,y1,x2,y2,x3,y3,x4,y4]
    xs = q[0::2]
    ys = q[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]

def draw_debug(image: Image.Image, bboxes=None, quads=None, out_path="debug.png", max_draw=80, labels=None):
    im = image.copy().convert("RGB")
    draw = ImageDraw.Draw(im)

    if bboxes:
        for i, b in enumerate(bboxes[:max_draw]):
            x1,y1,x2,y2 = map(int, b)
            draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
            tag = labels[i] if labels and i < len(labels) else f"b{i}"
            draw.text((x1, max(0,y1-14)), tag, fill="red")

    if quads:
        for i, q in enumerate(quads[:max_draw]):
            pts = [(q[0],q[1]), (q[2],q[3]), (q[4],q[5]), (q[6],q[7])]
            draw.polygon(pts, outline="lime", width=3)
            x1,y1,x2,y2 = map(int, quad_to_bbox(q))
            draw.text((x1, max(0,y1-14)), f"q{i}", fill="lime")

    im.save(out_path)

def bbox_center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def dist(p,q):
    return math.hypot(p[0]-q[0], p[1]-q[1])

def pick_by_priors(boxes, W, H, max_pick=3, alpha=0.08, beta=0.08):
    """
    你的先验：水印1在中心，水印2在左上，水印3在右下（偏移比例 alpha,beta 可调）
    boxes: list of [x1,y1,x2,y2]
    """
    cx, cy = W/2.0, H/2.0
    dx, dy = alpha*W, beta*H
    priors = [(cx, cy), (cx-dx, cy-dy), (cx+dx, cy+dy)]

    picked = []
    used = set()
    for pr in priors[:max_pick]:
        best_i, best_d = None, 1e18
        for i, b in enumerate(boxes):
            if i in used:
                continue
            c = bbox_center(b)
            d = dist(c, pr)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is not None:
            used.add(best_i)
            picked.append(boxes[best_i])
    return picked

def extract_bboxes(task_out, task_key, W, H):
    if not isinstance(task_out, dict):
        return []
    raw = task_out.get(task_key, {})
    if not isinstance(raw, dict):
        return []
    bboxes = raw.get("bboxes", []) or []
    clean = []
    for b in bboxes:
        if len(b) == 4:
            clean.append(clamp_bbox(b, W, H))
    return clean

def extract_ovd(task_out, task_key, W, H):
    """Return (bboxes, labels, scores) from <OPEN_VOCABULARY_DETECTION>."""
    if not isinstance(task_out, dict):
        return [], [], []
    raw = task_out.get(task_key, {})
    if not isinstance(raw, dict):
        return [], [], []
    bboxes = raw.get("bboxes", []) or []
    labels = raw.get("labels", []) or []
    scores = raw.get("scores", []) or []
    clean = []
    for b in bboxes:
        if len(b) == 4:
            clean.append(clamp_bbox(b, W, H))
    return clean, labels, scores

def main(img_path: str, out_json: str):
    image = Image.open(img_path).convert("RGB")
    W, H = image.size

    ovd = run_task(image, "<OPEN_VOCABULARY_DETECTION>", "watermark; logo; signature; stamp")
    region = run_task(image, "<REGION_PROPOSAL>")
    ocr = run_task(image, "<OCR_WITH_REGION>")

    # 取出原始 boxes
    region_boxes = extract_bboxes(region, "<REGION_PROPOSAL>", W, H)
    quad_boxes = ocr.get("<OCR_WITH_REGION>", {}).get("quad_boxes", []) if isinstance(ocr, dict) else []
    ocr_boxes = [quad_to_bbox(q) for q in quad_boxes]
    ocr_boxes = [clamp_bbox(b, W, H) for b in ocr_boxes]
    ovd_boxes, ovd_labels, ovd_scores = extract_ovd(ovd, "<OPEN_VOCABULARY_DETECTION>", W, H)

    # 把“所有候选”画出来（你用这两张图先判断模型有没有碰到水印）
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    draw_debug(image, bboxes=region_boxes, out_path=out_json.replace(".json", "_region_all.png"))
    draw_debug(image, quads=quad_boxes, out_path=out_json.replace(".json", "_ocr_all.png"))
    if ovd_boxes:
        draw_debug(image, bboxes=ovd_boxes, labels=ovd_labels, out_path=out_json.replace(".json", "_ovd_all.png"))

    # 用先验筛 1~3 个“最像水印位置”的框（优先用 OVD，其次 OCR，最后 region）
    candidate_boxes = ovd_boxes if len(ovd_boxes) > 0 else (ocr_boxes if len(ocr_boxes) > 0 else region_boxes)
    picked = pick_by_priors(candidate_boxes, W, H, max_pick=3, alpha=0.08, beta=0.08)

    # 把 picked 也画出来（你最关心这个）
    draw_debug(image, bboxes=picked, out_path=out_json.replace(".json", "_picked.png"))

    out = {
        "image": img_path,
        "open_vocab_detection": ovd,
        "region_proposal": region,
        "ocr_with_region": ocr,
        "picked_boxes": picked,
        "picked_from": "ovd_boxes" if len(ovd_boxes) > 0 else ("ocr_boxes" if len(ocr_boxes) > 0 else "region_boxes")
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("saved:", out_json)
    print("debug images:",
          out_json.replace(".json", "_region_all.png"),
          out_json.replace(".json", "_ocr_all.png"),
          out_json.replace(".json", "_picked.png"))
    



    

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.img, args.out)
