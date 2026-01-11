import os
import random
import cv2


DATASET_DIR = "data/dataset"
ORB_NFEATURES = 1000


def load_images(dataset_dir):
    images = []
    image_paths = []

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                images.append(img)
                image_paths.append(path)

    return images, image_paths


def extract_orb_features(images, nfeatures=ORB_NFEATURES):
    orb = cv2.ORB_create(nfeatures=nfeatures)

    keypoints_list = []
    descriptors_list = []

    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        keypoints_list.append(kp)
        descriptors_list.append(des)

    return keypoints_list, descriptors_list


if __name__ == "__main__":
    # Load dataset
    images, paths = load_images(DATASET_DIR)
    print(f"[INFO] Loaded {len(images)} images")

    # Extract ORB features
    keypoints, descriptors = extract_orb_features(images)
    print("[INFO] ORB extraction done")

    # Print a few statistics
    for i in range(3):
        if keypoints[i] is not None:
            print(f"{paths[i]} -> {len(keypoints[i])} keypoints")
        else:
            print(f"{paths[i]} -> 0 keypoints")

    # ---- Visualization (sanity check) ----
    idx = random.randint(0, len(images) - 1)

    if keypoints[idx] is not None and len(keypoints[idx]) > 0:
        img_kp = cv2.drawKeypoints(
            images[idx],
            keypoints[idx],
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        cv2.imshow("ORB keypoints", img_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("[WARN] Selected image has no keypoints to display.")
