import cv2
import time
import torch
# torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


def main():
    # 读入图像
    image = Image.open('/home/suma/projects/undercar_detection/undercar_datasets/images/train/20200820153147668495.jpg')
    image = np.array(image.convert("RGB"))
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"), apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        stability_score_offset=1.0,
        mask_threshold=0.0,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        point_grids=None,
        min_mask_region_area=0,
        output_mode="binary_mask",
        use_m2m=False,
        multimask_output=True)
    start_time = time.time()
    masks = mask_generator.generate(image)
    end_time = time.time()
    process_time = end_time - start_time
    print('process_time:', process_time)
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()