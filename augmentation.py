"""
augmentation.py
----------------
Augmentation Pipeline สำหรับ Classification ใบทุเรียน
รองรับ 3 ระดับ: none | v1 | v2

Design Rationale
----------------
ใบทุเรียนถ่ายในสภาพแสงและฉากหลัง (background) ที่หลากหลาย
การเพิ่ม Gaussian Noise และ Gaussian Blur ช่วยให้โมเดล:
  1. ทนต่อ sensor noise / JPEG compression artifacts ในภาพจริง
  2. เรียนรู้ feature ของใบ (เส้นใบ, สี, เนื้อผิว) โดยไม่ยึดติดกับความคมชัดของ background
     → เทียบเท่ากับการ "blur background" แบบ implicit ผ่าน noise regularization

หมายเหตุ: Augmentation ทำงานเฉพาะตอน training=True เท่านั้น
ไม่กระทบ Validation/Test dataset

Backend Compatibility:
- พัฒนาโดยใช้ Keras 3 (keras.ops / keras.random)
- รองรับทั้ง PyTorch (GPU/CUDA) และ TensorFlow backend โดยไม่มีปัญหา tensor device/type conversion
"""

import keras
from keras import layers, ops, random, Sequential
import numpy as np


# ---------------------------------------------------------------------------
# Gaussian Noise Layer (custom — backend-agnostic สำหรับ Keras 3 / PyTorch / TF)
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="durian_leaf")
class GaussianNoise(layers.Layer):
    """
    เพิ่ม Gaussian Noise บนภาพ (stddev ถูก scale ตาม pixel range 0-255)

    การทำงาน:
    - สุ่ม noise จาก N(0, stddev) แล้วบวกเข้าไปกับทุก pixel
    - ทำงานเฉพาะตอน training=True เพื่อไม่ให้กระทบ inference
    - clip ค่าให้อยู่ใน [0, 255] เสมอ
    - ช่วยให้โมเดลทนต่อ sensor noise, compression artifact และ background clutter

    Args:
        stddev: ค่าเบี่ยงเบนมาตรฐานของ noise (pixel scale 0–255)
    """
    def __init__(self, stddev: float = 8.0, **kwargs):
        super().__init__(**kwargs)
        self.stddev = float(stddev)

    def call(self, x, training=False):
        if training and self.stddev > 0.0:
            noise = random.normal(shape=ops.shape(x), mean=0.0, stddev=self.stddev, dtype=x.dtype)
            x = ops.clip(x + noise, 0.0, 255.0)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config


# ---------------------------------------------------------------------------
# Gaussian Blur Layer (custom — ใช้ depthwise convolution เพื่อ blur, backend-agnostic)
# ---------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="durian_leaf")
class GaussianBlur(layers.Layer):
    """
    ใช้ Gaussian Kernel ผ่าน depthwise convolution เพื่อ blur ภาพ

    การทำงาน:
    - สร้าง Gaussian kernel ขนาด kernel_size x kernel_size
    - ใช้ depthwise_conv กับทุก channel พร้อมกัน
    - ทำงานเฉพาะตอน training=True
    - เป็นการจำลอง "depth of field" เพื่อให้โมเดลโฟกัสที่ feature ของใบ
      แทนที่จะจำ background texture

    Args:
        kernel_size: ขนาด kernel (ควรเป็นเลขคี่) — ยิ่งมาก ยิ่ง blur มาก
        sigma:       ค่าความกว้างของ Gaussian curve
        apply_prob:  ความน่าจะเป็นที่จะ apply blur ต่อ 1 ภาพ (random apply)
    """
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0, apply_prob: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.sigma = float(sigma)
        self.apply_prob = float(apply_prob)
        self.kernel = None

    @staticmethod
    def _build_kernel(size: int, sigma: float, channels: int):
        """สร้าง 2D Gaussian kernel ที่ normalize แล้วในรูป NumPy [K, K, channels, 1]"""
        ax = np.arange(size) - size // 2
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = (kernel / np.sum(kernel)).astype("float32")
        kernel = np.expand_dims(kernel, axis=(-1, -2))  # [K, K, 1, 1]
        kernel = np.repeat(kernel, channels, axis=2)     # [K, K, channels, 1]
        return kernel

    def build(self, input_shape):
        channels = input_shape[-1]
        kernel_np = self._build_kernel(self.kernel_size, self.sigma, channels)
        self.kernel = self.add_weight(
            name="gaussian_kernel",
            shape=kernel_np.shape,
            initializer=keras.initializers.Constant(kernel_np),
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, x, training=False):
        if not training or self.apply_prob <= 0.0:
            return x

        ndim = len(ops.shape(x))
        x_in = ops.expand_dims(x, axis=0) if ndim == 3 else x
        channels = ops.shape(x_in)[-1]

        kernel = self.kernel
        if kernel is None:
            kernel_np = self._build_kernel(self.kernel_size, self.sigma, channels)
            kernel = ops.convert_to_tensor(kernel_np, dtype=x.dtype)

        blurred = ops.depthwise_conv(x_in, kernel, strides=1, padding="same")
        batch_size = ops.shape(x_in)[0]
        probs = random.uniform(shape=(batch_size, 1, 1, 1))
        res = ops.where(probs < self.apply_prob, blurred, x_in)
        return ops.squeeze(res, axis=0) if ndim == 3 else res

    def get_config(self):
        config = super().get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "sigma": self.sigma,
            "apply_prob": self.apply_prob,
        })
        return config


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_augmentation(version: str):
    """
    สร้าง augmentation pipeline ตาม version ที่กำหนด

    Args:
        version: 'none' | 'v1' | 'v2'

    Returns:
        keras.Sequential หรือ None (กรณี 'none')

    Augmentation Design:
    ┌─────┬────────────────────────────────────────────────────────────┐
    │ v1  │ Mild — สำหรับชุดข้อมูลขนาดปานกลาง                        │
    │     │ · RandomFlip, Rotation ±3%, Zoom 5%, Translation 5%       │
    │     │ · GaussianNoise (stddev=6): จำลอง sensor noise             │
    │     │ · GaussianBlur (σ=0.8, p=0.3): blur background เบาๆ       │
    ├─────┼────────────────────────────────────────────────────────────┤
    │ v2  │ Moderate — สำหรับชุดข้อมูลน้อย / oversampled data         │
    │     │ · RandomFlip, Rotation ±6%, Zoom 10%, Translation 8%      │
    │     │ · Brightness 10%, Contrast 10%                             │
    │     │ · GaussianNoise (stddev=12): noise หนักขึ้น               │
    │     │ · GaussianBlur (σ=1.2, p=0.5): blur background ชัดขึ้น   │
    └─────┴────────────────────────────────────────────────────────────┘
    """
    version = version.lower()

    if version == "none":
        return None

    if version == "v1":
        # Mild Augmentation + Light Noise/Blur
        return Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.03),           # ±~11°
                layers.RandomZoom(0.05),               # zoom ±5%
                layers.RandomTranslation(0.05, 0.05),  # shift ±5%
                GaussianNoise(stddev=6.0),             # sensor noise จำลอง
                GaussianBlur(kernel_size=3, sigma=0.8, apply_prob=0.3),  # blur background เบาๆ
            ],
            name="augmentation_v1_mild",
        )

    if version == "v2":
        # Moderate Augmentation + Stronger Noise/Blur
        return Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.06),           # ±~22°
                layers.RandomZoom(0.10),               # zoom ±10%
                layers.RandomTranslation(0.08, 0.08),  # shift ±8%
                layers.RandomBrightness(0.10),
                layers.RandomContrast(0.10),
                GaussianNoise(stddev=12.0),            # noise หนักขึ้น ทน compression
                GaussianBlur(kernel_size=5, sigma=1.2, apply_prob=0.5),  # blur background ชัดเจน
            ],
            name="augmentation_v2_moderate",
        )

    raise ValueError(f"Unknown augmentation version: '{version}'. Choices: none, v1, v2")
