import argparse
import pathlib
from sys import builtin_module_names
import cv2
import numpy as np
import tqdm
# import emnist
import os
import tensorflow as tf
import tensorflow_datasets as tfds

def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.
    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = prediction_box
    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    # Compute intersection
    x1i = max(x1_t, x1_p)
    x2i = min(x2_t, x2_p)
    y1i = max(y1_t, y1_p)
    y2i = min(y2_t, y2_p)
    intersection = (x2i - x1i) * (y2i - y1i)

    # Compute union
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_t - x1_t) * (y2_t - y1_t)
    union = pred_area + gt_area - intersection
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def compute_iou_all(bbox, all_bboxes):
    ious = [0]
    for other_bbox in all_bboxes:
        ious.append(
            calculate_iou(bbox, other_bbox)
        )
    return ious


def tight_bbox(digit, orig_bbox):
    xmin, ymin, xmax, ymax = orig_bbox
    # xmin
    shift = 0
    for i in range(digit.shape[1]):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmin += shift
    # xmax
    shift = 0
    for i in range(-1, -digit.shape[1], -1):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmax -= shift
    # ymin
    shift = 0
    for i in range(digit.shape[0]):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymin += shift
    shift = 0
    for i in range(-1, -digit.shape[0], -1):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymax -= shift
    return [xmin, ymin, xmax, ymax]


def dataset_exists(dirpath: pathlib.Path, num_images):
    if not dirpath.is_dir():
        return False
    for image_id in range(num_images):
        error_msg = f"EMNIST Detection dataset already generated in {dirpath}, \n\tbut did not find filepath:"
        error_msg2 = f"You can delete the directory by running: rm -r {dirpath.parent}"
        impath = dirpath.joinpath("images", f"{image_id}.png")
        assert impath.is_file(), f"{error_msg} {impath} \n\t{error_msg2}"
        label_path = dirpath.joinpath("labels", f"{image_id}.txt")
        assert label_path.is_file(),  f"{error_msg} {impath} \n\t{error_msg2}"
    return True

def generate_dataset(dirpath: pathlib.Path,
                     num_images: int,
                     max_digit_size: int,
                     min_digit_size: int,
                    #  imsize: int,
                     max_digits_per_image: int,
                     emnist_letters_images: np.ndarray,
                     emnist_letters_labels: np.ndarray,
                     coco_ds: tf.data.Dataset):
    # if dataset_exists(dirpath, num_images):
    #     return
    max_image_value = 255
    assert emnist_letters_images.dtype == np.uint8
    # assert coco_ds.dtype == np.uint8
    image_dir = dirpath.joinpath("images")
    label_dir = dirpath.joinpath("labels")
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    coco_ds = tuple(coco_ds.take(num_images))
    for image_id in tqdm.trange(num_images, desc=f"Generating dataset, saving to: {dirpath}"):
        bg_idx = np.random.randint(0, len(coco_ds))
        bg_image = coco_ds[bg_idx]["image"]
        bg_width, bg_height, _ = bg_image.shape
        im = np.zeros((bg_width, bg_height), dtype=np.float32)
        labels = []
        bboxes = []
        num_images = np.random.randint(1, max_digits_per_image)
        for _ in range(num_images):
            while True:
                width = np.random.randint(min_digit_size, max_digit_size)
                if bg_width-width <= 0 or bg_height-width <= 0:
                    continue

                x0 = np.random.randint(0, bg_width-width)
                y0 = np.random.randint(0, bg_height-width)
                ious = compute_iou_all([x0, y0, x0+width, y0+width], bboxes)
                if max(ious) < 0.1:
                    break
            # while True:
            digit_idx = np.random.randint(0, len(emnist_letters_images))
            digit = emnist_letters_images[digit_idx].astype(np.float32)
            digit = cv2.resize(digit, (width, width))
            label = emnist_letters_labels[digit_idx]
            labels.append(label)
            assert im[x0:x0+width, y0:y0+width].shape == digit.shape, f"imshape: {im[x0:x0+width, y0:y0+width].shape}, digit shape: {digit.shape}"
            bbox = tight_bbox(digit, [y0, x0, y0+width, x0+width])
            bboxes.append(bbox)

            im[x0:x0+width, y0:y0+width] += digit
            im[im > max_image_value] = max_image_value
        
        # while True:
        #     bg_idx = np.random.randint(0, len(coco_ds))
        #     bg_image = coco_ds[bg_idx]["image"]
        #     if bg_image.shape[0] >= 300 and bg_image.shape[1] >= 300:
        #         break

        bg_image = tfds.as_numpy(bg_image)
        
            

        image_target_path = image_dir.joinpath(f"{image_id}.jpg")
        label_target_path = label_dir.joinpath(f"{image_id}.txt")

        # Randomized pixel image
        # bg_image = np.random.random((300, 300, 3)) * 255

        im = 255 - im
        im = np.stack((im,) * 3, axis=-1)
        im = np.where(im >= 200, bg_image, im)
        im = cv2.GaussianBlur(im, (3,3), 0)
        im = im.astype(np.uint8)
        is_success, im_buf_arr = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        if is_success:
            im_buf_arr.tofile(str(image_target_path))

            with open(label_target_path, "w") as fp:
                fp.write("label,xmin,ymin,xmax,ymax\n")
                for l, bbox in zip(labels, bboxes):
                    bbox = [str(_) for _ in bbox]
                    to_write = f"{l}," + ",".join(bbox) + "\n"
                    fp.write(to_write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path", default="dataset/emnist_letters_detection"
    )
    # parser.add_argument(
    #     "--imsize", default=300, type=int
    # )
    parser.add_argument(
        "--max-digit-size", default=200, type=int
    )
    parser.add_argument(
        "--min-digit-size", default=30, type=int
    )
    parser.add_argument(
        "--num-train-images", default=10000, type=int
    )
    parser.add_argument(
        "--num-test-images", default=1000, type=int
    )
    parser.add_argument(
        "--max-digits-per-image", default=5, type=int
    )
    parser.add_argument(
        "--subset", default='letters', type=str
    )
    args = parser.parse_args()

    emnist_letters_train_image, emnist_letters_train_label = tfds.as_numpy(tfds.load(
        'emnist/letters',
        split='train',
        batch_size=-1,
        as_supervised=True,
        shuffle_files=True,
        data_dir="d:/tensorflow_dataset/"
    ))

    emnist_letters_test_image, emnist_letters_test_label = tfds.as_numpy(tfds.load(
        'emnist/letters',
        split='test',
        batch_size=-1,
        as_supervised=True,
        shuffle_files=True,
        data_dir="d:/tensorflow_dataset/"
    ))

    coco_train_ds = tfds.load(
        'coco/2017',
        split='train',
        data_dir="d:/tensorflow_dataset/",
        shuffle_files=True
    )
    assert isinstance(coco_train_ds, tf.data.Dataset)

    coco_test_ds = tfds.load(
        'coco/2017',
        split='test',
        data_dir="d:/tensorflow_dataset/",
        shuffle_files=True,
    )
    assert isinstance(coco_test_ds, tf.data.Dataset)

    train_dataset = [emnist_letters_train_image, emnist_letters_train_label, coco_train_ds]
    test_dataset = [emnist_letters_test_image, emnist_letters_test_label, coco_test_ds]

    for dataset, (target_images, target_labels, bg_images) in zip(["train", "test"], [train_dataset, test_dataset]):
        num_images = args.num_train_images if dataset == "train" else args.num_test_images
        generate_dataset(
            pathlib.Path(args.base_path, dataset),
            num_images,
            args.max_digit_size,
            args.min_digit_size,
            # args.imsize,
            args.max_digits_per_image,
            target_images,
            target_labels,
            bg_images
        )
