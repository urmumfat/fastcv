import time
import os
import cv2
import torch
import fastcv
import numpy as np

def benchmark_hist_eq(image_path, runs=50):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    img_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_np is None:
        print("Error reading image.")
        return

    img_torch = torch.from_numpy(img_np).cuda()

    print(f"\n=== Benchmarking Histogram Equalization ===")
    print(f"Image Resolution: {img_np.shape}")

    start = time.perf_counter()
    for _ in range(runs):
        _ = cv2.equalizeHist(img_np)
    end = time.perf_counter()
    cv_time = (end - start) / runs * 1000

    fastcv.histogram_equalization(img_torch)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = fastcv.histogram_equalization(img_torch)
    torch.cuda.synchronize()
    end = time.perf_counter()
    fc_time = (end - start) / runs * 1000

    print(f"OpenCV (CPU): {cv_time:.4f} ms")
    print(f"fastcv (CUDA): {fc_time:.4f} ms")
    print(f"Speedup: {cv_time / fc_time:.2f}x")

    result_tensor = fastcv.histogram_equalization(img_torch)
    result_np = result_tensor.cpu().numpy()
    cv2.imwrite("../artifacts/output_gpu_eq.jpg", result_np)

if __name__ == "__main__":
    img_path = "../artifacts/images.jpg" 
    benchmark_hist_eq(img_path)