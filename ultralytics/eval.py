from ultralytics import YOLO
# from wandb.integration.ultralytics import add_wandb_callback
import wandb
import yaml
import glob
# Create a new YOLO model
model = YOLO("/home/hugo/ultralytics/ultralytics/yolov8-small-kitti/train2/weights/best.pt")
# model.eval()
# model.val(data="/home/hugo/datasets/voc/dataset.yaml", plots=True, save_json=True, batch=4, imgsz=800)
# for dataset_name in ["ID-bdd-OOD-coco", "OOD-open", "voc-ood"]:
# # for dataset_name in ["ID-voc-OOD-coco"]:
# # for dataset_name in ["OOD-open","voc-ood"]:
    # print(f"evaluation for {dataset_name}:")
    # model.val(data=f"/home/hugo/datasets/{dataset_name}/dataset.yaml", plots=True, save_json=True, batch=4, imgsz=800)
model.val(data="/home/hugo/fiftyone/kitti/KITTI/dataset.yaml", plots=True, save_json=True)    
# export model to onnx
# model.export(format='onnx', half=True)