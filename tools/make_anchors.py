#!/usr/bin/env python3
import pandas as pd
import numpy as np

def bbox_iou(box, clusters):
    """
    Calculate IoU between a single box and cluster anchors.
    box: [w, h], clusters: [[w1, h1], [w2, h2], ...]
    """
    w_min = np.minimum(clusters[:, 0], box[0])
    h_min = np.minimum(clusters[:, 1], box[1])
    inter = w_min * h_min
    area_box = box[0] * box[1]
    area_clusters = clusters[:, 0] * clusters[:, 1]
    return inter / (area_box + area_clusters - inter + 1e-9)


def kmeans_iou(wh, k, seed=42, max_iter=300):
    """
    Perform IoU-based KMeans clustering on width-height array wh into k clusters.
    Returns cluster centers and assignments.
    """
    np.random.seed(seed)
    # Initialize clusters as k random boxes
    clusters = wh[np.random.choice(wh.shape[0], k, replace=False)].astype(float)
    last_assignments = None

    for _ in range(max_iter):
        # Compute distance = 1 - IoU
        distances = np.array([1 - bbox_iou(box, clusters) for box in wh])
        assignments = np.argmin(distances, axis=1)
        # If assignments didn't change, convergence
        if last_assignments is not None and np.all(assignments == last_assignments):
            break
        # Update clusters to median of assigned boxes
        for i in range(k):
            if np.any(assignments == i):
                clusters[i] = np.median(wh[assignments == i], axis=0)
        last_assignments = assignments

    return clusters, assignments


def main():
    # Path to the CSV of bounding boxes
    CSV_PATH = "/home/jeongwoo/AUE8088/kaist_rgbt_bboxes.csv"
    NUM_ANCHORS = 9

    # Load CSV and filter out ignored annotations
    df = pd.read_csv(CSV_PATH)
    df = df[df['ignore'] == 0]

    # Extract width and height in pixels
    wh = df[['bbox_w', 'bbox_h']].to_numpy()

    # Perform IoU KMeans to get anchors
    anchors, _ = kmeans_iou(wh, NUM_ANCHORS)

    # Sort anchors by area and reshape into 3 scales of 3 each
    order = np.argsort(anchors[:, 0] * anchors[:, 1])
    anchors = anchors[order].reshape(3, 3, 2)

    # Print in YOLO YAML format
    print("anchors:")
    scales = ['P3/8', 'P4/16', 'P5/32']
    for i, scale in enumerate(scales):
        vals = ", ".join(f"{int(w)},{int(h)}" for w, h in anchors[i])
        print(f"  - [{vals}]  # {scale}")


if __name__ == "__main__":
    main()
