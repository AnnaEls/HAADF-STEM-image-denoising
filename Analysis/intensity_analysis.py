import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from Analysis.gaussian_detection import detect_and_plot_gaussian_blobs, fit_2d_gaussians, gaussian_2d

def cluster_gaussians_by_amplitude_and_plot(
    calibrated_img_array: np.ndarray,
    fitted_gaussians: list,
    n_clusters: int = 3,
    random_state: int = 42,
    figsize=(12, 10),
    verbose: bool = True
):
    """
    Cluster fitted Gaussians by amplitude using K-Means and plot the
    clustered Gaussian centers over the calibrated image.

    Parameters
    ----------
    calibrated_img_array : np.ndarray
        2D calibrated image.
    fitted_gaussians : list of dict
        Output from Gaussian fitting. Each dict must contain
        ['amplitude', 'x0', 'y0'].
    n_clusters : int
        Number of K-Means clusters.
    random_state : int
        Random seed for reproducibility.
    figsize : tuple
        Figure size for plotting.
    verbose : bool
        Print status messages.

    Returns
    -------
    cluster_labels : np.ndarray
        Cluster label for each Gaussian.
    kmeans : KMeans
        Fitted KMeans object.
    """

    if calibrated_img_array is None:
        raise ValueError("calibrated_img_array is None.")

    if fitted_gaussians is None or len(fitted_gaussians) == 0:
        raise ValueError("fitted_gaussians is empty.")

    # Extract amplitudes (shape: Nx1 for sklearn)
    gaussian_amplitudes = np.array(
        [[g['amplitude']] for g in fitted_gaussians]
    )

    # K-Means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init='auto'
    )

    cluster_labels = kmeans.fit_predict(gaussian_amplitudes)

    if verbose:
        print(f"K-Means clustering completed for {len(cluster_labels)} Gaussians.")
        print("First 10 cluster labels:", cluster_labels[:10])

    # ---------------- FINAL OVERLAY PLOT ----------------
    plt.figure(figsize=figsize)
    plt.imshow(calibrated_img_array, cmap='gray')

    # Use a colormap for arbitrary cluster count
    cmap = plt.cm.get_cmap('tab10', n_clusters)

    for cluster_id in range(n_clusters):
        xs = [
            fitted_gaussians[i]['x0']
            for i, lbl in enumerate(cluster_labels)
            if lbl == cluster_id
        ]
        ys = [
            fitted_gaussians[i]['y0']
            for i, lbl in enumerate(cluster_labels)
            if lbl == cluster_id
        ]

        plt.scatter(
            xs,
            ys,
            s=100,
            color=cmap(cluster_id),
            edgecolors='black',
            linewidth=1,
            label=f'Cluster {cluster_id}'
        )

    plt.title('Gaussian Centers Clustered by Amplitude')
    plt.axis('off')
    plt.legend()
    plt.show()

    return cluster_labels, kmeans

def build_gaussian_training_data(
    fitted_gaussians: list,
    cluster_labels: np.ndarray,
    return_dataframe: bool = True
):
    """
    Convert fitted Gaussian parameters and cluster labels into
    training-ready feature and target arrays.

    Parameters
    ----------
    fitted_gaussians : list of dict
        Each dict must contain:
        ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'offset']
    cluster_labels : np.ndarray
        Cluster label for each Gaussian.
    return_dataframe : bool
        If True, returns X as a Pandas DataFrame.
        If False, returns X as a NumPy array.

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape (N, 6).
    y : np.ndarray
        Target vector of shape (N,).
    """

    if fitted_gaussians is None or len(fitted_gaussians) == 0:
        return None, None

    if cluster_labels is None or len(cluster_labels) == 0:
        raise ValueError("cluster_labels is empty or None.")

    if len(fitted_gaussians) != len(cluster_labels):
        raise ValueError(
            "Length mismatch: fitted_gaussians and cluster_labels must match."
        )

    # Feature extraction
    feature_names = ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'offset']

    X_data = np.array([
        [
            g['amplitude'],
            g['x0'],
            g['y0'],
            g['sigma_x'],
            g['sigma_y'],
            g['offset']
        ]
        for g in fitted_gaussians
    ])

    y = np.asarray(cluster_labels)

    if return_dataframe:
        X = pd.DataFrame(X_data, columns=feature_names)
    else:
        X = X_data

    return X, y

def train_random_forest_classifier(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth=None,
    verbose: bool = True
):
    """
    Train a Random Forest classifier and report evaluation metrics.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape (N, F).
    y : np.ndarray
        Target labels of shape (N,).
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int or None
        Maximum tree depth.
    verbose : bool
        Print evaluation metrics.

    Returns
    -------
    model : RandomForestClassifier
        Trained classifier.
    metrics : dict
        Dictionary containing evaluation metrics.
    """

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }

    if verbose:
        print("=== Random Forest Classification Metrics ===")
        print(f"Accuracy        : {metrics['accuracy']:.4f}")
        print(f"Precision (avg) : {metrics['precision_macro']:.4f}")
        print(f"Recall (avg)    : {metrics['recall_macro']:.4f}")
        print(f"F1-score (avg)  : {metrics['f1_macro']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])
        print("\nClassification Report:")
        print(metrics["classification_report"])

    return model, metrics

import numpy as np
import pandas as pd

def build_gaussian_prediction_data(
    fitted_gaussians: list,
    return_dataframe: bool = True
):
    """
    Convert fitted Gaussian parameters into dataset for preciciotn.

    Parameters
    ----------
    fitted_gaussians : list of dict
        Each dict must contain:
        ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'offset']
    return_dataframe : bool
        If True, returns X as a Pandas DataFrame.
        If False, returns X as a NumPy array.

    Returns
    -------
    X : pd.DataFrame or np.ndarray
        Feature matrix of shape (N, 6).
    """

    if fitted_gaussians is None or len(fitted_gaussians) == 0:
        return None

    # Feature extraction
    feature_names = ['amplitude', 'x0', 'y0', 'sigma_x', 'sigma_y', 'offset']

    X_data = np.array([
        [
            g['amplitude'],
            g['x0'],
            g['y0'],
            g['sigma_x'],
            g['sigma_y'],
            g['offset']
        ]
        for g in fitted_gaussians
    ])

    if return_dataframe:
        X = pd.DataFrame(X_data, columns=feature_names)
    else:
        X = X_data

    return X

def classification_metrics_to_dataframe(
    y_true,
    y_pred,
    labels,
    *,
    total_gaussians=None,
    matched=None,
    extra_metadata=None,
    zero_division=0,
):
    """
    Convert classification results into a single-row pandas DataFrame.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    labels : list
        Ordered list of class labels (e.g. [0, 1, 2]).
    total_gaussians : int, optional
        Total number of evaluated objects.
    matched : int, optional
        Number of valid / matched objects.
    extra_metadata : dict, optional
        Additional experiment metadata (e.g. dose, noise, model name).
    zero_division : int, default=0
        Passed to sklearn.classification_report.

    Returns
    -------
    df : pandas.DataFrame
        Single-row DataFrame with all metrics flattened.
    """

    # --- classification report ---
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=zero_division,
    )

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row = {}

    # --- global metrics ---
    row["accuracy"] = report["accuracy"]

    if total_gaussians is not None:
        row["total_gaussians"] = total_gaussians
    if matched is not None:
        row["matched"] = matched
        row["misclassified"] = matched - int(round(report["accuracy"] * matched))

    # --- per-class metrics ---
    for cls in labels:
        cls_key = str(cls)
        row[f"class_{cls}_precision"] = report[cls_key]["precision"]
        row[f"class_{cls}_recall"]    = report[cls_key]["recall"]
        row[f"class_{cls}_f1"]        = report[cls_key]["f1-score"]
        row[f"class_{cls}_support"]   = report[cls_key]["support"]

    # --- macro & weighted averages ---
    for avg in ("macro avg", "weighted avg"):
        key = avg.replace(" ", "_")
        row[f"{key}_precision"] = report[avg]["precision"]
        row[f"{key}_recall"]    = report[avg]["recall"]
        row[f"{key}_f1"]        = report[avg]["f1-score"]

    # --- confusion matrix (flattened) ---
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            row[f"cm_true{t}_pred{p}"] = cm[i, j]

    # --- extra metadata (dose, noise, model, seed, etc.) ---
    if extra_metadata is not None:
        row.update(extra_metadata)

    return pd.DataFrame([row])


def to_numpy_labels(y):
      if isinstance(y, (pd.Series, pd.DataFrame)):
          return y.squeeze().to_numpy()
      return np.asarray(y)

def classify_intensities(img, model, X, y, verbose=False):
  gaussians = detect_and_plot_gaussian_blobs(img,threshold=0.2,show=verbose)
  fitted_gaussians = fit_2d_gaussians(img, gaussians, gaussian_2d,verbose=verbose)
  # Check if fitted_gaussians is empty or None before proceeding
  if not fitted_gaussians:
    return None
  X_ns = build_gaussian_prediction_data(fitted_gaussians)
  if len(X_ns) == 0:
    return None
  y_ns = model.predict(X_ns[['amplitude']])


  pos_cols = ['x0', 'y0']     # Gaussian center columns
  max_dist = 6            # distance threshold

  point_size = 80

  y_np     = to_numpy_labels(y)
  y_ns_np = to_numpy_labels(y_ns)

  # ---- Extract positions ----
  X_pos    = X[pos_cols].to_numpy()
  X_ns_pos = X_ns[pos_cols].to_numpy()

  # ---- Hungarian matching ----
  C = cdist(X_ns_pos, X_pos, metric="euclidean")
  row_ind, col_ind = linear_sum_assignment(C)

  matched_dist = C[row_ind, col_ind]
  valid = matched_dist < max_dist

  # ---- Matched labels ----
  y_true = y_np[col_ind[valid]]
  y_pred = y_ns_np[row_ind[valid]]

  # ---- Metrics ----
  acc = accuracy_score(y_true, y_pred)

  result = classification_metrics_to_dataframe(y_true,
                                               y_pred,
                                               labels=[0,1,2],
                                               total_gaussians=len(X_ns),
                                               matched=valid.sum())



  if verbose:
    cm  = confusion_matrix(y_true, y_pred)
    print("===== METRICS ===")
    print(f"Total Gaussians (X_ns): {len(X_ns)}")
    print(f"Matched (valid):        {valid.sum()}")
    print(f"Accuracy:               {acc:.4f}\n")

    print("Confusion Matrix:")
    print(cm, "\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # ---- Identify misclassifications ----
    mis_mask = (y_true != y_pred)
    mis_idx  = np.where(mis_mask)[0]

    print(f"Misclassified points: {len(mis_idx)}")

    # ---- Coordinates for plotting ----
    matched_pos = X_ns_pos[row_ind[valid]]
    mis_pos     = matched_pos[mis_mask]


    # ---- Plot ----
    plt.figure(figsize=(8, 8))
    h, w = img.shape[:2]
    plt.imshow(img, cmap="gray", extent=[0, w, h, 0])
    plt.axis("off")

    # Plot correctly classified points
    plt.scatter(
        matched_pos[~mis_mask, 0],
        matched_pos[~mis_mask, 1],
        c=y_pred[~mis_mask],
        cmap="tab10",
        s=point_size,
        edgecolors="k",
        linewidths=0.5,
        label="Correct"
    )

    # Plot misclassified points (red cross)
    plt.scatter(
        mis_pos[:, 0],
        mis_pos[:, 1],
        marker="x",
        c="red",
        s=point_size * 1.2,
        linewidths=2,
        label="Misclassified"
    )

    #plt.title(f"Dose {dose} e/A2")
    plt.legend(loc="lower center", ncol=2, frameon=True)
    plt.tight_layout()
    plt.show()
  return result, y_true, y_pred, row_ind, X_ns_pos, 
