#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt


ROOT = Path(__file__).resolve().parent
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LOGGER = trt.Logger(trt.Logger.WARNING)


class CudaRuntime:
    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

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

    def copy(self, destination, source, size, direction):
        self.check(
            self.lib.cudaMemcpy(destination, source, size, direction),
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


def softmax(values):
    values = values.astype(np.float64)
    values -= np.max(values)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def infer(engine_path, image_path, labels):
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"无法加载引擎: {engine_path}")
    context = engine.create_execution_context()

    input_index = next(i for i in range(engine.num_bindings) if engine.binding_is_input(i))
    output_index = next(i for i in range(engine.num_bindings) if not engine.binding_is_input(i))
    context.set_binding_shape(input_index, (1, 3, 224, 224))

    input_data = preprocess(image_path)
    output_shape = tuple(context.get_binding_shape(output_index))
    output_dtype = trt.nptype(engine.get_binding_dtype(output_index))
    output_data = np.empty(output_shape, dtype=output_dtype)

    cuda = CudaRuntime()
    device_input = cuda.malloc(input_data.nbytes)
    device_output = cuda.malloc(output_data.nbytes)
    bindings = [0] * engine.num_bindings
    bindings[input_index] = int(device_input.value)
    bindings[output_index] = int(device_output.value)

    try:
        cuda.copy(
            device_input,
            ctypes.c_void_p(input_data.ctypes.data),
            input_data.nbytes,
            cuda.HOST_TO_DEVICE,
        )
        for _ in range(10):
            if not context.execute_v2(bindings):
                raise RuntimeError("TensorRT 推理执行失败")

        start = time.perf_counter()
        if not context.execute_v2(bindings):
            raise RuntimeError("TensorRT 推理执行失败")
        cuda.copy(
            ctypes.c_void_p(output_data.ctypes.data),
            device_output,
            output_data.nbytes,
            cuda.DEVICE_TO_HOST,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    finally:
        cuda.free(device_input)
        cuda.free(device_output)

    logits = output_data.reshape(-1)
    probabilities = softmax(logits)
    order = np.argsort(probabilities)[::-1]

    print(f"图片: {image_path}")
    print(f"引擎: {engine_path}")
    print(f"单次推理耗时（含传回结果）: {elapsed_ms:.3f} ms")
    print("预测结果:")
    for index in order:
        name = labels[index] if index < len(labels) else f"class_{index}"
        print(f"  {name:<20} {probabilities[index] * 100:7.3f}%")
    print(f"最终类别: {labels[order[0]]}")


def main():
    parser = argparse.ArgumentParser(description="MobileNetV4 TensorRT 单图推理")
    parser.add_argument("image", help="待识别图片路径")
    parser.add_argument(
        "--engine",
        default=str(ROOT / "best_model_int8.engine"),
        help="TensorRT 引擎路径",
    )
    args = parser.parse_args()

    labels_path = ROOT / "classes.txt"
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 5:
        raise RuntimeError(f"classes.txt 应包含 5 类，当前为 {len(labels)} 类")
    infer(Path(args.engine), Path(args.image), labels)


if __name__ == "__main__":
    main()
