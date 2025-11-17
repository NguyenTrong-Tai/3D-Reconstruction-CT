import modal
import subprocess
import sys

app = modal.App("dmcvr-experiments")

# # 1. Volume chứa data ACDC (preprocessed)
vol_data = modal.Volume.from_name("dmcvr-data", create_if_missing=True)

# # (Tuỳ bạn: đoạn upload này có thể tách ra file riêng, nhưng để đơn giản vẫn ok)
# with vol_data.batch_upload() as batch:
#     batch.put_directory(
#         "../datasets/ACDC_2D",
#         "/ACDC_2D"
#     )

# 2. Image: clone repo + cài CBIM
image = (
    modal.Image
    .debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.2.0",
        "torchvision==0.17.0",
        "torchaudio==2.2.0",
        "numpy",
        "scipy",
        "tqdm",
        "einops",
        "nibabel",
        "SimpleITK",
        "pydicom",
        "opencv-python",
        "pyyaml",
        "torchinfo", 
    )
    .run_commands(
        "mkdir -p /workspace",
        "cd /workspace && git clone https://github.com/NguyenTrong-Tai/3D-Reconstruction-CT.git",
        # Cài requirements của CBIM bên trong repo của bạn
        "cd /workspace/3D-Reconstruction-CT/segmentation/CBIM-Medical-Image-Segmentation && pip install -r requirement.txt",
    )
    .workdir("/workspace/3D-Reconstruction-CT/segmentation/CBIM-Medical-Image-Segmentation")
)

GPU_TYPE = "A100"

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/data": vol_data},
    timeout=60 * 60 * 12,  # 12h
)
def train_acdc(
    dataset: str = "acdc",
    model: str = "medformer",
    dimension: str = '2d',
    gpu: int = 0,
    batch_size: int = 32,
):
    # Tạo symlink dataset từ Volume /data/acdc -> repo/datasets/ACDC_2D
    # (nếu đã tồn tại thì check=False để tránh lỗi)
    subprocess.run(
        [
            "ln",
            "-s",
            "/data/acdc",
            "datasets/ACDC_2D",
        ],
        check=False,
    )

    cmd = [
        "python",
        "train.py",
        "--dataset",
        dataset,
        "--model",
        model,
        "--dimension",
        dimension,
        "--gpu",
        str(gpu),
        "--batch_size",
        str(batch_size),
    ]

    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)


@app.local_entrypoint()
def main():
    dataset = "acdc"
    model = "medformer"
    dimension = '2d'
    gpu = 0  # nên dùng 0
    batch_size = 32

    train_acdc.remote(dataset, model, dimension, gpu, batch_size)
