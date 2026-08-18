from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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


# ============================================================
# CONFIG
# ============================================================

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY")

STORAGE_BUCKET = "tomato-images"

INPUT_W = 640
INPUT_H = 640

CONF_THRESHOLD = 0.10
IOU_THRESHOLD = 0.45


if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL environment variable not found")

if not SUPABASE_SECRET_KEY:
    raise RuntimeError("SUPABASE_SECRET_KEY environment variable not found")


# ตัด / ท้าย URL เผื่อใส่มา
SUPABASE_URL = SUPABASE_URL.rstrip("/")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Tomato AI Backend",
    version="3.0.0"
)


# ============================================================
# ONNX RUNTIME
# ============================================================

options = ort.SessionOptions()

# ลด resource สำหรับ Render Free
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1

options.enable_mem_pattern = False
options.enable_cpu_mem_arena = False


session = ort.InferenceSession(
    "best.onnx",
    sess_options=options,
    providers=[
        "CPUExecutionProvider"
    ]
)


input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

input_name = input_info.name


print("ONNX loaded")
print("Input:", input_info.name, input_info.shape)
print("Output:", output_info.name, output_info.shape)


# ============================================================
# CLASS NAMES
# ============================================================

def get_class_names():

    try:

        metadata = (
            session
            .get_modelmeta()
            .custom_metadata_map
        )

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

        print(
            "Cannot read class metadata:",
            e
        )


    # fallback ถ้า metadata หาย
    return {
        0: "class_0",
        1: "class_1",
        2: "class_2",
        3: "class_3",
        4: "class_4",
        5: "class_5",
        6: "class_6",
        7: "class_7",
        8: "class_8",
        9: "class_9",
        10: "class_10",
        11: "class_11",
        12: "class_12",
        13: "class_13",
        14: "class_14",
        15: "class_15"
    }


CLASS_NAMES = get_class_names()

print("Classes:", CLASS_NAMES)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess(image: Image.Image):

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


    # YOLO letterbox background
    canvas = Image.new(
        "RGB",
        (INPUT_W, INPUT_H),
        (114, 114, 114)
    )


    pad_x = (
        INPUT_W - new_w
    ) // 2

    pad_y = (
        INPUT_H - new_h
    ) // 2


    canvas.paste(
        resized,
        (pad_x, pad_y)
    )


    image_array = np.asarray(
        canvas,
        dtype=np.float32
    )


    # 0 - 255 -> 0 - 1
    image_array /= 255.0


    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (2, 0, 1)
    )


    # CHW -> BCHW
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    image_array = np.ascontiguousarray(
        image_array,
        dtype=np.float32
    )


    return (
        image_array,
        ratio,
        pad_x,
        pad_y,
        original_w,
        original_h
    )


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box,
    boxes
):

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


    intersection_width = np.maximum(
        0,
        x2 - x1
    )

    intersection_height = np.maximum(
        0,
        y2 - y1
    )


    intersection = (
        intersection_width
        *
        intersection_height
    )


    area_box = (
        (box[2] - box[0])
        *
        (box[3] - box[1])
    )


    area_boxes = (
        (boxes[:, 2] - boxes[:, 0])
        *
        (boxes[:, 3] - boxes[:, 1])
    )


    union = (
        area_box
        +
        area_boxes
        -
        intersection
        +
        1e-6
    )


    return (
        intersection
        /
        union
    )


# ============================================================
# NMS
# ============================================================

def nms(
    boxes,
    scores,
    class_ids,
    iou_threshold=0.45
):

    keep = []


    unique_classes = np.unique(
        class_ids
    )


    for class_id in unique_classes:

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


# ============================================================
# YOLO POSTPROCESS
#
# ONNX:
# INPUT  [1,3,640,640]
# OUTPUT [1,20,8400]
#
# 20 =
# 4 bbox
# +
# 16 class score
# ============================================================

def postprocess(
    output,
    ratio,
    pad_x,
    pad_y,
    original_w,
    original_h,
    conf_threshold=CONF_THRESHOLD,
    iou_threshold=IOU_THRESHOLD
):

    prediction = np.squeeze(
        output
    )


    # [20,8400]
    # ->
    # [8400,20]

    if (
        prediction.ndim == 2
        and
        prediction.shape[0]
        <
        prediction.shape[1]
    ):

        prediction = prediction.T


    if prediction.ndim != 2:

        raise RuntimeError(
            f"Unexpected model output: "
            f"{prediction.shape}"
        )


    # ==============================
    # bbox
    # ==============================

    boxes = prediction[:, :4]


    # ==============================
    # class probabilities
    # ==============================

    class_scores = (
        prediction[:, 4:]
    )


    class_ids = np.argmax(
        class_scores,
        axis=1
    )


    scores = np.max(
        class_scores,
        axis=1
    )


    # ==============================
    # Confidence filter
    # ==============================

    mask = (
        scores
        >=
        conf_threshold
    )


    boxes = boxes[mask]

    scores = scores[mask]

    class_ids = class_ids[mask]


    if len(boxes) == 0:
        return []


    # ========================================================
    # YOLO xywh -> xyxy
    # ========================================================

    xyxy = np.empty_like(
        boxes
    )


    xyxy[:, 0] = (
        boxes[:, 0]
        -
        boxes[:, 2] / 2
    )

    xyxy[:, 1] = (
        boxes[:, 1]
        -
        boxes[:, 3] / 2
    )

    xyxy[:, 2] = (
        boxes[:, 0]
        +
        boxes[:, 2] / 2
    )

    xyxy[:, 3] = (
        boxes[:, 1]
        +
        boxes[:, 3] / 2
    )


    # ========================================================
    # Remove letterbox padding
    # ========================================================

    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y


    # ========================================================
    # Scale กลับไปขนาดรูปเดิม
    # ========================================================

    xyxy /= ratio


    # ========================================================
    # จำกัด bbox ไม่ให้ออกนอกรูป
    # ========================================================

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


    # ========================================================
    # NMS
    # ========================================================

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


        confidence = float(
            scores[i]
        )


        x1, y1, x2, y2 = (
            xyxy[i]
        )


        result = {

            "class_id":
                class_id,

            "class_name":
                CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                ),

            "confidence":
                round(
                    confidence,
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
        }


        results.append(
            result
        )


    # confidence สูงสุดก่อน
    results.sort(
        key=lambda item:
            item["confidence"],
        reverse=True
    )


    return results


# ============================================================
# CONTENT TYPE
# ============================================================

def normalize_content_type(
    content_type
):

    valid_types = {

        "image/jpeg",
        "image/png",
        "image/webp"

    }


    if content_type in valid_types:

        return content_type


    # ESP32-CAM บางทีส่งมาไม่ระบุ
    return "image/jpeg"


# ============================================================
# FILE EXTENSION
# ============================================================

def get_extension(
    content_type
):

    if content_type == "image/png":
        return ".png"

    if content_type == "image/webp":
        return ".webp"

    return ".jpg"


# ============================================================
# SUPABASE STORAGE UPLOAD
# ============================================================

def upload_image_to_supabase(
    image_bytes,
    content_type,
    device_id
):

    content_type = (
        normalize_content_type(
            content_type
        )
    )


    extension = get_extension(
        content_type
    )


    now = datetime.now(
        timezone.utc
    )


    timestamp = now.strftime(
        "%Y%m%d_%H%M%S"
    )


    unique_id = (
        uuid.uuid4()
        .hex[:10]
    )


    # เช่น
    # esp32-cam-01/20260819_012300_abcd123.jpg

    image_path = (
        f"{device_id}/"
        f"{timestamp}_"
        f"{unique_id}"
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


    print(
        "Storage status:",
        response.status_code
    )


    if response.status_code not in (
        200,
        201
    ):

        print(
            "Storage response:",
            response.text
        )

        raise RuntimeError(
            f"Supabase Storage error "
            f"{response.status_code}: "
            f"{response.text}"
        )


    return image_path


# ============================================================
# PUBLIC IMAGE URL
# ============================================================

def get_public_image_url(
    image_path
):

    return (
        f"{SUPABASE_URL}"
        f"/storage/v1/object/public/"
        f"{STORAGE_BUCKET}/"
        f"{image_path}"
    )


# ============================================================
# SAVE DATABASE
# ============================================================

def save_detection(
    device_id,
    image_path,
    class_name,
    confidence
):

    url = (
        f"{SUPABASE_URL}"
        f"/rest/v1/"
        f"disease_detections"
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
            device_id,

        "image_path":
            image_path,

        "class_name":
            class_name,

        "confidence":
            float(confidence)
    }


    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=20
    )


    print(
        "Database status:",
        response.status_code
    )


    if response.status_code not in (
        200,
        201,
        204
    ):

        print(
            "Database response:",
            response.text
        )

        raise RuntimeError(
            f"Supabase Database error "
            f"{response.status_code}: "
            f"{response.text}"
        )


    return True


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status":
            "ok",

        "message":
            "Tomato AI Backend",

        "engine":
            "ONNX Runtime",

        "database":
            "Supabase",

        "storage":
            STORAGE_BUCKET,

        "confidence_threshold":
            CONF_THRESHOLD
    }


# ============================================================
# MODEL INFO
# ============================================================

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
            len(CLASS_NAMES),


        "confidence_threshold":
            CONF_THRESHOLD,


        "iou_threshold":
            IOU_THRESHOLD
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
async def predict(

    file: UploadFile = File(...),

    # ถ้าไม่ส่ง device_id
    # จะใช้ esp32-cam-01 อัตโนมัติ
    device_id: str = Form(
        "esp32-cam-01"
    )
):

    # ========================================================
    # 1. READ IMAGE
    # ========================================================

    try:

        raw = await file.read()


        if not raw:

            raise ValueError(
                "Empty image"
            )


        image = Image.open(
            io.BytesIO(raw)
        )


        # โหลดข้อมูลจริงจากไฟล์
        image.load()


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid image: {e}"
            )
        )


    print(
        "Received image:",
        file.filename
    )

    print(
        "Device:",
        device_id
    )

    print(
        "Image size:",
        image.size
    )

    print(
        "Bytes:",
        len(raw)
    )


    # ========================================================
    # 2. PREPROCESS
    # ========================================================

    try:

        (
            tensor,
            ratio,
            pad_x,
            pad_y,
            original_w,
            original_h
        ) = preprocess(
            image
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Preprocess error: {e}"
            )
        )


    # ========================================================
    # 3. ONNX INFERENCE
    # ========================================================

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

        print(
            "Inference error:",
            e
        )


        raise HTTPException(
            status_code=500,
            detail=(
                f"AI inference error: {e}"
            )
        )


    print(
        "Detection count:",
        len(detections)
    )


    # ========================================================
    # 4. UPLOAD ORIGINAL IMAGE TO SUPABASE
    # ========================================================

    try:

        content_type = (
            normalize_content_type(
                file.content_type
            )
        )


        image_path = (
            upload_image_to_supabase(

                image_bytes=raw,

                content_type=
                    content_type,

                device_id=
                    device_id
            )
        )


        image_url = (
            get_public_image_url(
                image_path
            )
        )


    except Exception as e:

        print(
            "Storage upload error:",
            e
        )


        raise HTTPException(
            status_code=500,
            detail=(
                f"Storage upload error: {e}"
            )
        )


    # ========================================================
    # 5. CHOOSE BEST RESULT
    # ========================================================

    if len(detections) > 0:

        best = detections[0]


        class_id = (
            best["class_id"]
        )


        class_name = (
            best["class_name"]
        )


        confidence = (
            best["confidence"]
        )


    else:

        class_id = None

        class_name = (
            "No_Detection"
        )

        confidence = 0.0


    # ========================================================
    # 6. SAVE DATABASE
    #
    # บันทึกทุกภาพ
    # แม้ AI ไม่เจอโรค
    # ========================================================

    database_saved = False

    database_error = None


    try:

        save_detection(

            device_id=
                device_id,

            image_path=
                image_path,

            class_name=
                class_name,

            confidence=
                confidence
        )


        database_saved = True


    except Exception as e:

        database_error = str(e)


        print(
            "Database save error:",
            database_error
        )


    # ========================================================
    # 7. API RESPONSE
    # ========================================================

    return {

        "success":
            True,


        "device_id":
            device_id,


        "image": {

            "path":
                image_path,

            "url":
                image_url
        },


        "result": {

            "class_id":
                class_id,

            "class_name":
                class_name,

            "confidence":
                confidence
        },


        "database_saved":
            database_saved,


        "database_error":
            database_error,


        "count":
            len(detections),


        "detections":
            detections
    }