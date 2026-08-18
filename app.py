from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

import onnxruntime as ort
import numpy as np

import requests
import os
import io
import ast
import json
import uuid
from datetime import datetime, timezone


# =========================================================
# CONFIG
# =========================================================

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

STORAGE_BUCKET = "tomato-images"

DEVICE_ID = "esp32-cam-01"

INPUT_W = 640
INPUT_H = 640


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Tomato AI Backend",
    version="3.0"
)


# =========================================================
# ONNX RUNTIME
# =========================================================

options = ort.SessionOptions()

# ลด resource สำหรับ Render Free
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.enable_mem_pattern = False
options.enable_cpu_mem_arena = False

session = ort.InferenceSession(
    "best.onnx",
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

input_name = input_info.name


# =========================================================
# CLASS NAMES
# =========================================================

def get_class_names():

    try:

        metadata = session.get_modelmeta().custom_metadata_map

        raw_names = metadata.get("names")

        if raw_names:

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


    # ถ้า metadata ไม่มีชื่อ class
    return {
        i: f"class_{i}"
        for i in range(16)
    }


CLASS_NAMES = get_class_names()

print("Classes:", CLASS_NAMES)


# =========================================================
# IMAGE PREPROCESS
# =========================================================

def preprocess(image):

    image = image.convert("RGB")

    original_w, original_h = image.size

    ratio = min(
        INPUT_W / original_w,
        INPUT_H / original_h
    )

    new_w = int(
        round(original_w * ratio)
    )

    new_h = int(
        round(original_h * ratio)
    )


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


    array = np.asarray(
        canvas,
        dtype=np.float32
    )


    array /= 255.0


    # HWC -> CHW
    array = np.transpose(
        array,
        (2, 0, 1)
    )


    # CHW -> BCHW
    array = np.expand_dims(
        array,
        axis=0
    )


    array = np.ascontiguousarray(
        array,
        dtype=np.float32
    )


    return (
        array,
        ratio,
        pad_x,
        pad_y,
        original_w,
        original_h
    )


# =========================================================
# IOU
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
            np.argsort(
                scores[indices]
            )[::-1]
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
# YOLO POSTPROCESS
# output = [1,20,8400]
# =========================================================

def postprocess(
    output,
    ratio,
    pad_x,
    pad_y,
    original_w,
    original_h,
    conf_threshold=0.25,
    iou_threshold=0.45
):

    prediction = np.squeeze(output)


    # [20,8400] -> [8400,20]
    if (
        prediction.ndim == 2
        and prediction.shape[0]
        < prediction.shape[1]
    ):

        prediction = prediction.T


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

    xyxy = np.empty_like(boxes)


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


    # เอา letterbox padding ออก
    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y


    # scale กลับขนาดรูปจริง
    xyxy /= ratio


    # จำกัด bbox
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


    keep = nms(
        xyxy,
        scores,
        class_ids,
        iou_threshold
    )


    results = []


    for i in keep:

        class_id = int(
            class_ids[i]
        )

        x1, y1, x2, y2 = (
            xyxy[i]
        )


        results.append({

            "class_id":
                class_id,

            "class_name":
                CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                ),

            "confidence":
                round(
                    float(scores[i]),
                    4
                ),

            "bbox": {

                "x1":
                    int(round(x1)),

                "y1":
                    int(round(y1)),

                "x2":
                    int(round(x2)),

                "y2":
                    int(round(y2))
            }
        })


    results.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )


    return results


# =========================================================
# SUPABASE STORAGE
# =========================================================

def get_extension(content_type):

    if content_type == "image/png":
        return ".png"

    if content_type == "image/webp":
        return ".webp"

    return ".jpg"


def upload_image_to_supabase(
    image_bytes,
    content_type
):

    extension = get_extension(
        content_type
    )


    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )


    unique_id = uuid.uuid4().hex[:8]


    image_path = (
        f"{DEVICE_ID}/"
        f"{timestamp}_{unique_id}"
        f"{extension}"
    )


    url = (
        f"{SUPABASE_URL}"
        f"/storage/v1/object/"
        f"{STORAGE_BUCKET}/"
        f"{image_path}"
    )


    headers = {

        "apikey":
            SUPABASE_SECRET_KEY,

        "Content-Type":
            content_type,

        "x-upsert":
            "false"
    }


    response = requests.post(
        url,
        headers=headers,
        data=image_bytes,
        timeout=30
    )


    if response.status_code not in (
        200,
        201
    ):

        raise RuntimeError(
            "Storage error "
            f"{response.status_code}: "
            f"{response.text}"
        )


    return image_path


# =========================================================
# PUBLIC IMAGE URL
# =========================================================

def get_public_image_url(
    image_path
):

    return (
        f"{SUPABASE_URL}"
        f"/storage/v1/object/public/"
        f"{STORAGE_BUCKET}/"
        f"{image_path}"
    )


# =========================================================
# SAVE DETECTION TO DATABASE
# =========================================================

def save_detection(
    image_path,
    class_name,
    confidence
):

    url = (
        f"{SUPABASE_URL}"
        "/rest/v1/"
        "disease_detections"
    )


    headers = {

        "apikey":
            SUPABASE_SECRET_KEY,

        "Content-Type":
            "application/json",

        "Prefer":
            "return=minimal"
    }


    data = {

        "device_id":
            DEVICE_ID,

        "image_path":
            image_path,

        "class_name":
            class_name,

        "confidence":
            confidence
    }


    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=15
    )


    if response.status_code not in (
        200,
        201,
        204
    ):

        raise RuntimeError(
            "Database error "
            f"{response.status_code}: "
            f"{response.text}"
        )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {

        "status":
            "ok",

        "message":
            "Tomato AI Backend",

        "engine":
            "ONNX Runtime",

        "storage":
            "Supabase"
    }


# =========================================================
# MODEL INFO
# =========================================================

@app.get("/model-info")
def model_info():

    return {

        "input": {
            "name":
                input_info.name,

            "shape":
                input_info.shape
        },

        "output": {
            "name":
                output_info.name,

            "shape":
                output_info.shape
        },

        "classes":
            CLASS_NAMES,

        "class_count":
            len(CLASS_NAMES)
    }


# =========================================================
# PREDICT
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # =====================================================
    # อ่านรูป
    # =====================================================

    try:

        raw = await file.read()

        if not raw:

            raise ValueError(
                "Empty file"
            )


        image = Image.open(
            io.BytesIO(raw)
        )


        # บังคับให้ PIL โหลดข้อมูลจริง
        image.load()


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {e}"
        )


    # =====================================================
    # YOLO
    # =====================================================

    (
        tensor,
        ratio,
        pad_x,
        pad_y,
        original_w,
        original_h
    ) = preprocess(image)


    try:

        outputs = session.run(
            None,
            {
                input_name:
                    tensor
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


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"AI inference error: {e}"
        )


    # =====================================================
    # UPLOAD IMAGE
    # =====================================================

    content_type = (
        file.content_type
        or "image/jpeg"
    )


    try:

        image_path = (
            upload_image_to_supabase(
                raw,
                content_type
            )
        )


        image_url = (
            get_public_image_url(
                image_path
            )
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    # =====================================================
    # SAVE BEST DETECTION
    # =====================================================

    database_saved = False


    if len(detections) > 0:

        best = detections[0]


        try:

            save_detection(
                image_path=
                    image_path,

                class_name=
                    best["class_name"],

                confidence=
                    best["confidence"]
            )

            database_saved = True


        except Exception as e:

            # รูปอัปโหลดแล้ว แต่ DB ล้ม
            # ไม่จำเป็นต้องให้ API ล้มทั้งหมด
            print(
                "Database save error:",
                e
            )


    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "success":
            True,

        "device_id":
            DEVICE_ID,

        "image_path":
            image_path,

        "image_url":
            image_url,

        "database_saved":
            database_saved,

        "count":
            len(detections),

        "detections":
            detections
    }