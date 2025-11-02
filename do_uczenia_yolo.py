if __name__ == "__main__":
    import os
    import torch
    from ultralytics import YOLO
#do cuda
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({total_mem:.1f} GB)")
    else:
        device = "cpu"
        print("grafiki nie ma, robimy na procku")

    imgsz = 640
    batch_size = 8
    print(f"Using batch_size={batch_size}, imgsz={imgsz}, device={device}")
    model = YOLO("yolo11n.pt")
    model.train(
        data="do_uczenia/data.yaml",
        epochs=60,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        workers=0,
        cache=False,
        augment=True,
        project="runs/train",
        name="yolo11n_moje",
        verbose=True,
    )

    print("koniec")
