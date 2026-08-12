#!/usr/bin/env python3
import ctypes
import ctypes.util
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt


ROOT = Path(__file__).resolve().parent
ONNX_PATH = ROOT / "best_model.onnx"
IMAGE_DIR = ROOT / "calibration_images"
CACHE_PATH = ROOT / "mobilenetv4_int8.cache"
ENGINE_PATH = ROOT / "best_model_int8.engine"

INPUT_SHAPE = (1, 3, 224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOGGER = trt.Logger(trt.Logger.INFO)


class CudaRuntime:
    HOST_TO_DEVICE = 1

    def __init__(self):
        library = ctypes.util.find_library("cudart") or "libcudart.so"
        self.lib = ctypes.CDLL(library)
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.check(self.lib.cudaSetDevice(0), "cudaSetDevice")

    @staticmethod
    def check(code, operation):
        if code != 0:
            raise RuntimeError(f"{operation} failed, CUDA error code: {code}")

    def malloc(self, size):
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        return pointer

    def copy_to_device(self, destination, source):
        self.check(
            self.lib.cudaMemcpy(
                destination,
                ctypes.c_void_p(source.ctypes.data),
                source.nbytes,
                self.HOST_TO_DEVICE,
            ),
            "cudaMemcpy",
        )

    def free(self, pointer):
        if pointer and pointer.value:
            self.check(self.lib.cudaFree(pointer), "cudaFree")


def preprocess(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(image, dtype=np.float32)


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, cache_path):
        super().__init__()
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images = sorted(
            path for path in Path(image_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not self.images:
            raise RuntimeError(f"校准图片目录为空: {image_dir}")

        self.cache_path = Path(cache_path)
        self.index = 0
        self.cuda = CudaRuntime()
        self.device_input = self.cuda.malloc(np.prod(INPUT_SHAPE) * np.dtype(np.float32).itemsize)
        print(f"找到 {len(self.images)} 张校准图片")

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.index >= len(self.images):
            return None

        image_path = self.images[self.index]
        batch = preprocess(image_path)
        self.cuda.copy_to_device(self.device_input, batch)
        self.index += 1
        print(f"校准进度: {self.index}/{len(self.images)}", end="\r", flush=True)
        return [int(self.device_input.value)]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            print(f"读取已有校准缓存: {self.cache_path.name}")
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)
        print(f"\n已保存校准缓存: {self.cache_path.name}")

    def release(self):
        if getattr(self, "device_input", None):
            self.cuda.free(self.device_input)
            self.device_input = None


def main():
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"找不到模型: {ONNX_PATH}")
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"找不到校准图片目录: {IMAGE_DIR}")

    builder = trt.Builder(LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, LOGGER)

    if not parser.parse(ONNX_PATH.read_bytes()):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError("ONNX 模型解析失败")

    input_tensor = network.get_input(0)
    model_input_shape = tuple(input_tensor.shape)
    print(f"模型输入: {input_tensor.name} {model_input_shape}")
    if model_input_shape not in (INPUT_SHAPE, (-1, 3, 224, 224)):
        raise RuntimeError(
            f"输入形状不匹配，模型为 {model_input_shape}，期望固定或动态批次的 3x224x224"
        )
    if not builder.platform_has_fast_int8:
        raise RuntimeError("当前设备不支持快速 INT8")

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    else:
        config.max_workspace_size = 2 << 30
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    if model_input_shape[0] == -1:
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, INPUT_SHAPE, INPUT_SHAPE, INPUT_SHAPE)
        config.add_optimization_profile(profile)
        config.set_calibration_profile(profile)
        print("动态批次已固定为 1，用于 INT8 校准和推理")

    calibrator = ImageCalibrator(IMAGE_DIR, CACHE_PATH)
    config.int8_calibrator = calibrator
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT 创建 INT8 引擎失败")
        ENGINE_PATH.write_bytes(bytes(serialized_engine))
    finally:
        calibrator.release()

    print(f"INT8 引擎已生成: {ENGINE_PATH}")


if __name__ == "__main__":
    main()
