from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import onnxruntime as ort
import numpy as np
import io
import ast
import json

app = FastAPI()

# =========================================================
# ONNX Runtime - เน้นประหยัด RAM
# =========================================================

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1
opts.enable_mem_pattern = False
opts.enable_cpu_mem_arena = False

session = ort.InferenceSession(
    "best.onnx",
    sess_options=opts,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

INPUT_W = 640
INPUT_H = 640


# =========================================================
# อ่านชื่อ Class จาก metadata ของโมเดล
# =========================================================

def get_class_names():
    try:
        metadata = session.get_modelmeta().custom_metadata_map
        raw_names = metadata.get("names")

        if not raw_names:
            return {}

        try:
            names = json.loads(raw_names)
        except Exception:
            names = ast.literal_eval(raw_names)

        if isinstance(names, dict):
            return {
                int(k): str(v)
                for k, v in names.items()
            }

        if isinstance(names, list):
            return {
                i: str(v)
                for i, v in enumerate(names)
            }

    except Exception as e:
        print("Metadata error:", e)

    return {}


CLASS_NAMES = get_class_names()

print("Classes:", CLASS_NAMES)


# =========================================================
# Preprocess แบบ Letterbox
# =========================================================

def preprocess(image):

    image = image.convert("RGB")

    original_w, original_h = image.size

    ratio = min(
        INPUT_W / original_w,
        INPUT_H / original_h
    )

    new_w = int(round(original_w * ratio))
    new_h = int(round(original_h * ratio))

    resized = image.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (INPUT_W, INPUT_H),
        (114, 114, 114)
    )

    pad_x = (INPUT_W - new_w) // 2
    pad_y = (INPUT_H - new_h) // 2

    canvas.paste(
        resized,
        (pad_x, pad_y)
    )

    img = np.asarray(
        canvas,
        dtype=np.float32
    )

    img /= 255.0

    # HWC -> CHW
    img = np.transpose(
        img,
        (2, 0, 1)
    )

    # CHW -> BCHW
    img = np.expand_dims(
        img,
        axis=0
    )

    img = np.ascontiguousarray(
        img,
        dtype=np.float32
    )

    return (
        img,
        ratio,
        pad_x,
        pad_y,
        original_w,
        original_h
    )


# =========================================================
# IoU
# =========================================================

def calculate_iou(box, boxes):

    x1 = np.maximum(
        box[0],
        boxes[:, 0]
    )

    y1 = np.maximum(
        box[1],
        boxes[:, 1]
    )

    x2 = np.minimum(
        box[2],
        boxes[:, 2]
    )

    y2 = np.minimum(
        box[3],
        boxes[:, 3]
    )

    intersection = (
        np.maximum(0, x2 - x1)
        *
        np.maximum(0, y2 - y1)
    )

    area1 = (
        (box[2] - box[0])
        *
        (box[3] - box[1])
    )

    area2 = (
        (boxes[:, 2] - boxes[:, 0])
        *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = (
        area1
        + area2
        - intersection
        + 1e-6
    )

    return intersection / union


# =========================================================
# NMS
# =========================================================

def nms(
    boxes,
    scores,
    class_ids,
    iou_threshold=0.45
):

    keep = []

    for class_id in np.unique(class_ids):

        indices = np.where(
            class_ids == class_id
        )[0]

        order = indices[
            np.argsort(scores[indices])[::-1]
        ]

        while len(order) > 0:

            current = order[0]

            keep.append(current)

            if len(order) == 1:
                break

            remaining = order[1:]

            ious = calculate_iou(
                boxes[current],
                boxes[remaining]
            )

            order = remaining[
                ious <= iou_threshold
            ]

    return keep


# =========================================================
# Postprocess YOLO
# =========================================================

def postprocess(
    output,
    ratio,
    pad_x,
    pad_y,
    original_w,
    original_h,
    conf_threshold=0.25
):

    # [1,20,8400]
    prediction = np.squeeze(output)

    # [20,8400] -> [8400,20]
    prediction = prediction.T

    # 4 bbox + 16 class scores
    boxes = prediction[:, :4]

    class_scores = prediction[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1
    )

    scores = np.max(
        class_scores,
        axis=1
    )

    # Confidence filter
    mask = scores >= conf_threshold

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []


    # =====================================================
    # xywh -> xyxy
    # =====================================================

    xyxy = np.zeros_like(boxes)

    xyxy[:, 0] = (
        boxes[:, 0]
        - boxes[:, 2] / 2
    )

    xyxy[:, 1] = (
        boxes[:, 1]
        - boxes[:, 3] / 2
    )

    xyxy[:, 2] = (
        boxes[:, 0]
        + boxes[:, 2] / 2
    )

    xyxy[:, 3] = (
        boxes[:, 1]
        + boxes[:, 3] / 2
    )


    # =====================================================
    # เอา padding ออก
    # =====================================================

    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y

    xyxy /= ratio


    # จำกัดไม่ให้ออกนอกรูป
    xyxy[:, 0] = np.clip(
        xyxy[:, 0],
        0,
        original_w
    )

    xyxy[:, 1] = np.clip(
        xyxy[:, 1],
        0,
        original_h
    )

    xyxy[:, 2] = np.clip(
        xyxy[:, 2],
        0,
        original_w
    )

    xyxy[:, 3] = np.clip(
        xyxy[:, 3],
        0,
        original_h
    )


    # =====================================================
    # NMS
    # =====================================================

    keep = nms(
        xyxy,
        scores,
        class_ids
    )


    results = []

    for i in keep:

        class_id = int(class_ids[i])

        x1, y1, x2, y2 = xyxy[i]

        results.append({

            "class_id": class_id,

            "class_name": CLASS_NAMES.get(
                class_id,
                f"class_{class_id}"
            ),

            "confidence": round(
                float(scores[i]),
                4
            ),

            "bbox": {
                "x1": int(round(x1)),
                "y1": int(round(y1)),
                "x2": int(round(x2)),
                "y2": int(round(y2))
            }
        })


    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    return results


# =========================================================
# API
# =========================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "message": "Tomato AI Backend",
        "engine": "ONNX Runtime"
    }


@app.get("/model-info")
def model_info():

    return {
        "input": {
            "name": session.get_inputs()[0].name,
            "shape": session.get_inputs()[0].shape
        },

        "output": {
            "name": session.get_outputs()[0].name,
            "shape": session.get_outputs()[0].shape
        },

        "classes": CLASS_NAMES,

        "class_count": len(CLASS_NAMES)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:
        raw = await file.read()

        image = Image.open(
            io.BytesIO(raw)
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {e}"
        )


    (
        tensor,
        ratio,
        pad_x,
        pad_y,
        original_w,
        original_h
    ) = preprocess(image)


    outputs = session.run(
        None,
        {
            input_name: tensor
        }
    )


    detections = postprocess(
        outputs[0],
        ratio,
        pad_x,
        pad_y,
        original_w,
        original_h
    )


    return {
        "count": len(detections),
        "detections": detections
    }