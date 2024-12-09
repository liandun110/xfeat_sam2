import cv2
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from modules.xfeat import XFeat
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches


def visualize(img: np.ndarray, keypoints: torch.tensor):
    """Visualize feature points on the given image."""
    img_with_keypoints = img.copy()
    keypoints_np = keypoints.cpu().numpy() if isinstance(keypoints, torch.Tensor) else keypoints

    for point in keypoints_np:
        center = tuple(point.astype(int))
        cv2.circle(img_with_keypoints, center, 5, (0, 0, 255), thickness=-1)  # Red dots

    plt.figure(figsize=(8, 8))
    plt.imshow(img_with_keypoints[..., ::-1])
    plt.title("Visualized Keypoints")
    plt.axis("off")
    plt.show()


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        area = ann['area']
        bbox = ann['bbox']
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

            # 显示面积和边框位置
            center_x, center_y = int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2)
            ax.text(center_x, center_y, f"{int(area)}", color='yellow', fontsize=8, ha='center', va='center')

    ax.imshow(img)


def load_image(image_path: str):
    max_size = 1024
    image = Image.open(image_path)
    image = image.convert("RGB")
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int((new_width / width) * height)
    else:
        new_height = max_size
        new_width = int((new_height / height) * width)
    image = image.resize((new_width, new_height))
    image = np.array(image)
    return image


def get_unmatched_points(im1, im2):
    xfeat = XFeat()
    top_k = 1024
    img1 = xfeat.parse_input(im1)
    img2 = xfeat.parse_input(im2)
    out1 = xfeat.detectAndCompute(img1, top_k=top_k)[0]
    out2 = xfeat.detectAndCompute(img2, top_k=top_k)[0]
    idxs0, idxs1 = xfeat.match(out1['descriptors'], out2['descriptors'], min_cossim=-1)
    # 没匹配上的点的index
    unmatched_idxs0 = list(set(range(top_k)) - set(idxs0.cpu().numpy()))
    unmatched_idxs1 = list(set(range(top_k)) - set(idxs1.cpu().numpy()))

    # 没匹配上的点的坐标
    unmatched_mkpts_0 = out1['keypoints'][unmatched_idxs0].cpu().numpy()
    unmatched_mkpts_1 = out2['keypoints'][unmatched_idxs1].cpu().numpy()
    return unmatched_mkpts_0, unmatched_mkpts_1


def main():
    # torch默认设置
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 入读图像
    im1 = load_image('/home/suma/projects/undercar_detection/undercar_datasets/images/train/20200820153147668495.jpg')
    im2 = load_image('/home/suma/projects/undercar_detection/undercar_datasets/images/train/20200820153641660311.jpg')
    # im1 = load_image('/home/suma/Pictures/1.jpg')
    # im2 = load_image('/home/suma/Pictures/2.jpg')

    # 得到未匹配的点
    start_time = time.time()
    unmatched_points, _ = get_unmatched_points(im1, im2)
    end_time = time.time()
    process_time = end_time - start_time
    print('xfeat耗时:', process_time)

    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=torch.device("cuda"), apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=22,
        points_per_batch=512,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
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
    masks = mask_generator.generate(im1, unmatched_points)
    end_time = time.time()
    process_time = end_time - start_time
    print('process_time:', process_time)

    # 过滤面积过小或过大的区域
    masks = [x for x in masks if 200<x['area']<2000]

    # 可视化
    plt.figure(figsize=(20, 20))
    plt.imshow(im1)
    show_anns(masks)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()