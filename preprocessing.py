"""
preprocessing.py
-----------------
ใช้งานร่วมกันระหว่าง train_classifier.py, evaluation/evaluate_classifier.py และ app.py
เพื่อให้ Train / Validation / Test / Inference ใช้ preprocessing เดียวกันทุกจุด
ตามที่ระบุไว้ใน Section 10 ของ final_md

Pipeline: Preserve Aspect Ratio -> Resize -> Padding -> Model-specific preprocessing
"""

import numpy as np
from PIL import Image

from config import get_input_size


def resize_with_padding(image: Image.Image, target_size) -> Image.Image:
    """
    Resize ภาพโดยคงสัดส่วน (aspect ratio) แล้วเติม padding สีเทาให้ได้ขนาดสี่เหลี่ยม
    ตามที่กำหนดไว้ใน Section 10 (ห้ามบีบภาพจนรูปร่างใบผิดเพี้ยน)
    """
    target_h, target_w = target_size
    image = image.convert("RGB")
    src_w, src_h = image.size

    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, int(src_w * scale)), max(1, int(src_h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    # พื้นหลัง padding สีเทากลาง (114,114,114) ตามธรรมเนียมของงาน detection/classification
    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def get_preprocess_fn(model_name: str):
    """
    คืนฟังก์ชัน preprocess_input ที่ตรงกับ pretrained model แต่ละตัว
    (ต้องใช้ให้ตรงกับที่ pretrained weight ถูก train มา ไม่งั้น transfer learning
    จะได้ประโยชน์จาก pretrained weight ลดลง — ตามที่ระบุใน Section 10.1)
    """
    model_name = model_name.lower()
    
    def preprocess_caffe(x):
        import numpy as np
        # Caffe mode: RGB -> BGR and zero-center using ImageNet means
        if isinstance(x, np.ndarray):
            x = x[..., ::-1].astype(np.float32)
            mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)
            x -= mean
            return x
        else:
            import tensorflow as tf
            x = x[..., ::-1]
            mean = tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)
            x -= mean
            return x

    def preprocess_pass(x):
        return x

    if model_name in ["efficientnet_b0", "mobilenetv3_large"]:
        return preprocess_pass
    elif model_name == "resnet50":
        return preprocess_caffe
    else:
        raise ValueError(f"Unknown model_name for preprocessing: {model_name}")


def load_and_preprocess_image(image: Image.Image, model_name: str) -> np.ndarray:
    """
    รับภาพ (PIL.Image) คืน numpy array ที่พร้อมป้อนเข้าโมเดล shape = (1, H, W, 3)
    """
    target_size = get_input_size(model_name)
    padded = resize_with_padding(image, target_size)
    arr = np.array(padded).astype("float32")
    preprocess_fn = get_preprocess_fn(model_name)
    arr = preprocess_fn(arr)
    return np.expand_dims(arr, axis=0)
