import numpy as np

def silhouette_score(x, clusters):
    n_samples = x.shape[0]

    unique_clusters = np.unique(clusters)
    silhouette_values = np.zeros(n_samples)

    for i in range(n_samples):
        same_cluster = clusters == clusters[i]

        # intracluster distances (a)
        a = np.mean(np.linalg.norm(x[i] - x[same_cluster], axis=1)) if np.sum(same_cluster) > 1 else 0

        # intercluster distances (b)
        b = np.min([
            np.mean(np.linalg.norm(x[i] - x[clusters == label], axis=1))
            for label in unique_clusters if label != clusters[i]
        ])

        silhouette_values[i] = (b - a) / max(a, b)

    return np.mean(silhouette_values)

def davies_bouldin_score(x, clusters):
    unique_clusters = np.unique(clusters)
    n_clusters = unique_clusters.size

    # centroid
    cluster_means = np.array([
        np.mean(x[clusters == label], axis=0)
        for label in unique_clusters
    ])

    intra_cluster_distances = np.array([
        np.mean(np.linalg.norm(x[clusters == label] - cluster_means[i], axis=1))
        for i, label in enumerate(unique_clusters)
    ])

    db_values = []

    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                inter_cluster_distance = np.linalg.norm(cluster_means[i] - cluster_means[j])
                ratio = (intra_cluster_distances[i] + intra_cluster_distances[j]) / inter_cluster_distance
                max_ratio = max(max_ratio, ratio)
        db_values.append(max_ratio)

    return np.mean(db_values)

def v_measure_score(y, clusters):
    unique_y = np.unique(y)
    unique_clusters = np.unique(clusters)

    # ground truth entropy (H(Y))
    n_samples = y.size
    y_counts = np.array([np.sum(y == label) for label in unique_y])
    h_y = -np.sum((y_counts / n_samples) * np.log(y_counts / n_samples))

    # clusters entropy (H(C))
    cluster_counts = np.array([np.sum(clusters == cluster) for cluster in unique_clusters])
    h_c = -np.sum((cluster_counts / n_samples) * np.log(cluster_counts / n_samples))

    # mutual information (I(Y;C))
    mi = 0
    for label in unique_y:
        for cluster in unique_clusters:
            intersection_count = np.sum((y == label) & (clusters == cluster))
            if intersection_count > 0:
                mi += (intersection_count / n_samples) * np.log(
                    (intersection_count / n_samples) /
                    ((np.sum(y == label) / n_samples) * (np.sum(clusters == cluster) / n_samples))
                )

    homogeneity = mi / h_y if h_y > 0 else 1.0
    completeness = mi / h_c if h_c > 0 else 1.0
    v_measure = 2 * (homogeneity * completeness) / (homogeneity + completeness) if (homogeneity + completeness) > 0 else 0.0

    return v_measure

def fowlkes_mallows_score(y, clusters):
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            same_true = y[i] == y[j]
            same_cluster = clusters[i] == clusters[j]

            if same_true and same_cluster:
                tp += 1
            elif not same_true and same_cluster:
                fp += 1
            elif same_true and not same_cluster:
                fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    fms = np.sqrt(precision * recall) if precision + recall > 0 else 0

    return fms
