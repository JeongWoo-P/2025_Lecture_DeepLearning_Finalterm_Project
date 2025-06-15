#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def bbox_iou(box, clusters):
    w_min = np.minimum(clusters[:, 0], box[0])
    h_min = np.minimum(clusters[:, 1], box[1])
    inter = w_min * h_min
    area_box = box[0] * box[1]
    area_clusters = clusters[:, 0] * clusters[:, 1]
    return inter / (area_box + area_clusters - inter + 1e-9)


def kmeans_iou(wh, k, seed=42, max_iter=300):
    np.random.seed(seed)
    clusters = wh[np.random.choice(wh.shape[0], k, replace=False)].astype(float)
    last_assignments = None

    for _ in range(max_iter):
        distances = np.array([1 - bbox_iou(box, clusters) for box in wh])
        assignments = np.argmin(distances, axis=1)
        if last_assignments is not None and np.all(assignments == last_assignments):
            break
        for i in range(k):
            if np.any(assignments == i):
                clusters[i] = np.median(wh[assignments == i], axis=0)
        last_assignments = assignments

    return clusters, assignments


def plot_clusters(wh, anchors, assignments):
    colors = plt.cm.get_cmap("tab10", len(anchors))
    plt.figure(figsize=(10, 8))

    for i in range(len(anchors)):
        cluster_points = wh[assignments == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    s=20, label=f"Cluster {i+1}", color=colors(i), alpha=0.6)

    plt.scatter(anchors[:, 0], anchors[:, 1], s=200, c='black', marker='X', label='Anchors')
    plt.title("IoU-Based K-Means Clustering of Bounding Boxes")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    CSV_PATH = "/home/jeongwoo/AUE8088/kaist_rgbt_bboxes.csv"
    NUM_ANCHORS = 9

    df = pd.read_csv(CSV_PATH)
    df = df[df['ignore'] == 0]

    wh = df[['bbox_w', 'bbox_h']].to_numpy()

    anchors, assignments = kmeans_iou(wh, NUM_ANCHORS)

    # Sort and reshape
    order = np.argsort(anchors[:, 0] * anchors[:, 1])
    anchors_sorted = anchors[order].reshape(3, 3, 2)

    print("anchors:")
    scales = ['P3/8', 'P4/16', 'P5/32']
    for i, scale in enumerate(scales):
        vals = ", ".join(f"{int(w)},{int(h)}" for w, h in anchors_sorted[i])
        print(f"  - [{vals}]  # {scale}")

    # Visualization
    plot_clusters(wh, anchors, assignments)


if __name__ == "__main__":
    main()
