import json
import os
import sys
import time
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

FRAME_EXTRACT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FRAME_EXTRACT_DIR)
SAM3D_BODY_DIR = os.path.join(PROJECT_ROOT, "sam-3d-body")
if SAM3D_BODY_DIR not in sys.path:
    sys.path.insert(0, SAM3D_BODY_DIR)

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from sam_3d_body.metadata.mhr70 import mhr_names
from tools.build_detector import HumanDetector
from tools.vis_utils import visualize_joints_together


# Set folder paths
##############################################################################
video_name= "walking"

IMAGE_FOLDER = os.path.join(FRAME_EXTRACT_DIR, "output_frames", video_name)

GENERAL_OUTPUT_FOLDER = os.path.join(FRAME_EXTRACT_DIR, "outputs")

GENERAL_OVERLAY_OUTPUT_FOLDER = os.path.join(GENERAL_OUTPUT_FOLDER, "overlays")
OVERLAY_OUTPUT_FOLDER = os.path.join(GENERAL_OVERLAY_OUTPUT_FOLDER, video_name)

GENERAL_JOINT_OUTPUT_FOLDER = os.path.join(GENERAL_OUTPUT_FOLDER, "joints_npz")
JOINT_OUTPUT_FOLDER = os.path.join(GENERAL_JOINT_OUTPUT_FOLDER, video_name)
##############################################################################

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT, "checkpoints", "sam-3d-body-dinov3", "model.ckpt"
)
MHR_PATH = os.path.join(
    PROJECT_ROOT, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_TYPE = "body"
DETECTOR_NAME = "vitdet"
DETECTOR_INPUT_SIZE = 1024
BBOX_THRESH = 0.8
SAVE_OVERLAY = True

SELECTED_JOINTS = [
"nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
    "left-big-toe-tip",
    "left-small-toe-tip",
    "left-heel",
    "right-big-toe-tip",
    "right-small-toe-tip",
    "right-heel",
    "right-wrist",
    "left-wrist",
    "left-olecranon",
    "right-olecranon",
    "left-cubital-fossa",
    "right-cubital-fossa",
    "left-acromion",
    "right-acromion",
    "neck",
]

JOINT_NAME_TO_INDEX = {name: idx for idx, name in enumerate(mhr_names)}
SELECTED_JOINT_INDICES = {
    joint_name: JOINT_NAME_TO_INDEX[joint_name] for joint_name in SELECTED_JOINTS
}


def list_images(image_folder):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
    return sorted(
        image_path
        for ext in image_extensions
        for image_path in glob(os.path.join(image_folder, ext))
    )


def ensure_paths():
    if not os.path.isdir(IMAGE_FOLDER):
        raise FileNotFoundError(f"Input image folder not found: {IMAGE_FOLDER}")
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.isfile(MHR_PATH):
        raise FileNotFoundError(f"MHR asset not found: {MHR_PATH}")

    os.makedirs(GENERAL_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(GENERAL_OVERLAY_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(GENERAL_JOINT_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(OVERLAY_OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(JOINT_OUTPUT_FOLDER, exist_ok=True)


def build_estimator():
    device = torch.device("cuda" if DEVICE == "cuda" else "cpu")
    model, model_cfg = load_sam_3d_body(
        CHECKPOINT_PATH,
        device=device,
        mhr_path=MHR_PATH,
    )
    human_detector = HumanDetector(name=DETECTOR_NAME, device=device, path="")
    return SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=None,
    )


def save_selected_joints_npz(npz_path, outputs, image_path):
    people_joint_data = {}
    bboxes = []
    for person_id, person_output in enumerate(outputs):
        keypoints_3d = np.asarray(person_output["pred_keypoints_3d"], dtype=np.float32)
        selected = np.stack(
            [keypoints_3d[SELECTED_JOINT_INDICES[name]] for name in SELECTED_JOINTS],
            axis=0,
        ).astype(np.float32)
        people_joint_data[f"person_{person_id}"] = selected
        bboxes.append(np.asarray(person_output["bbox"], dtype=np.float32))

    np.savez_compressed(
        npz_path,
        image_name=os.path.basename(image_path),
        joint_names=np.asarray(SELECTED_JOINTS),
        bboxes=np.asarray(bboxes, dtype=np.float32),
        **people_joint_data,
    )


def main():
    ensure_paths()
    estimator = build_estimator()
    image_paths = list_images(IMAGE_FOLDER)

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {IMAGE_FOLDER}")

    metadata = {
        "image_folder": IMAGE_FOLDER,
        "overlay_output_folder": OVERLAY_OUTPUT_FOLDER,
        "joint_output_folder": JOINT_OUTPUT_FOLDER,
        "checkpoint_path": CHECKPOINT_PATH,
        "mhr_path": MHR_PATH,
        "device": DEVICE,
        "inference_type": INFERENCE_TYPE,
        "detector_name": DETECTOR_NAME,
        "detector_input_size": DETECTOR_INPUT_SIZE,
        "bbox_thresh": BBOX_THRESH,
        "selected_joints": SELECTED_JOINTS,
    }
    with open(
        os.path.join(JOINT_OUTPUT_FOLDER, "_metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    for image_path in tqdm(image_paths):
        t0 = time.perf_counter()
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=BBOX_THRESH,
            detector_input_size=DETECTOR_INPUT_SIZE,
            inference_type=INFERENCE_TYPE,
        )
        t1 = time.perf_counter()

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        npz_path = os.path.join(JOINT_OUTPUT_FOLDER, f"{image_name}.npz")
        save_selected_joints_npz(npz_path, outputs, image_path)

        if SAVE_OVERLAY:
            img = cv2.imread(image_path)
            overlay = visualize_joints_together(img, outputs)
            overlay_path = os.path.join(OVERLAY_OUTPUT_FOLDER, f"{image_name}.jpg")
            cv2.imwrite(overlay_path, overlay.astype(np.uint8))

        t2 = time.perf_counter()
        print(
            f"[timing] {os.path.basename(image_path)} "
            f"inference={t1 - t0:.2f}s "
            f"save={t2 - t1:.2f}s "
            f"total={t2 - t0:.2f}s"
        )


if __name__ == "__main__":
    main()
