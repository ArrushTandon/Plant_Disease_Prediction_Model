import os
import numpy as np
import cv2
import time
import pickle
import tensorflow as tf
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------- Reproducibility --------------------
np.random.seed(42)
tf.random.set_seed(42)

# Mixed precision only if GPU exists
if tf.config.list_physical_devices('GPU'):
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')

# -------------------- LBP --------------------
def compute_lbp(image, P=8, R=1):
    image = (image * 255).astype(np.uint8)
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# -------------------- DATA LOADING --------------------
def load_and_preprocess_image(path, target_size=(128, 128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Invalid image: {path}")
    img = cv2.resize(img, target_size)
    img = img / 255.0
    lbp = compute_lbp(img)
    img = np.expand_dims(img, axis=-1)
    return img, lbp

def load_dataset(data_dir, categories):
    images, lbp_features, labels = [], [], []
    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        print(f"Loading: {path}")
        for file in os.listdir(path):
            fpath = os.path.join(path, file)
            if not os.path.isfile(fpath):
                continue
            try:
                img, lbp = load_and_preprocess_image(fpath)
                images.append(img)
                lbp_features.append(lbp)
                labels.append(label)
            except Exception as e:
                print("Skipping:", fpath, e)
    print("Total images:", len(images))
    return np.array(images), np.array(lbp_features), np.array(labels)

# -------------------- CNN FEATURE EXTRACTOR --------------------
def build_cnn(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten()
    ])

# -------------------- SMA FEATURE SELECTION --------------------
def sma_feature_selection(X, y, iterations=25, population_size=20, alpha=0.9):
    n_samples, n_features = X.shape
    population = np.random.rand(population_size, n_features)

    best_solution = None
    best_fitness = -np.inf
    eps = 1e-8

    for iteration in range(iterations):
        fitness_scores = []

        for i in range(population_size):
            mask = population[i] > 0.5
            if mask.sum() == 0:
                fitness_scores.append(0)
                continue

            X_sel = X[:, mask]
            mi = mutual_info_classif(X_sel, y)
            relevance = np.mean(mi)
            sparsity = 1 - (X_sel.shape[1] / n_features)
            score = alpha * relevance + (1 - alpha) * sparsity
            fitness_scores.append(score)

            if score > best_fitness:
                best_fitness = score
                best_solution = population[i].copy()

        fitness_scores = np.array(fitness_scores)
        idx = np.argsort(fitness_scores)[::-1]
        population = population[idx]
        fitness_scores = fitness_scores[idx]

        f_best = fitness_scores[0]
        f_worst = fitness_scores[-1]

        W = np.zeros(population_size)
        for i in range(population_size):
            ratio = (f_best - fitness_scores[i]) / (f_best - f_worst + eps)
            if i < population_size // 2:
                W[i] = 1 + np.random.rand() * np.log(ratio + 1)
            else:
                W[i] = 1 - np.random.rand() * np.log(ratio + 1)

        for i in range(population_size):
            if np.random.rand() < 0.5:
                population[i] = best_solution + W[i] * (
                    np.random.rand(n_features) * best_solution -
                    np.random.rand(n_features) * population[i]
                )
            else:
                r = np.random.randint(population_size)
                population[i] = population[r] + W[i] * (
                    np.random.rand(n_features) * population[r] -
                    np.random.rand(n_features) * population[i]
                )

            population[i] = np.clip(population[i], 0, 1)

        print(f"[SMA] Iter {iteration+1}/{iterations} | Best fitness: {best_fitness:.4f}")

    if best_solution is None or (best_solution > 0.5).sum() == 0:
        print("WARNING: Empty mask — using all features")
        return np.ones(n_features, dtype=bool)

    return best_solution > 0.5

# -------------------- FINAL CLASSIFIER --------------------
def build_final_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# -------------------- MAIN --------------------
def main():
    data_dir = "data/processed"
    categories = ["healthy", "diseased"]

    images, lbp, labels = load_dataset(data_dir, categories)

    X_tr, X_te, lbp_tr, lbp_te, y_tr, y_te = train_test_split(
        images, lbp, labels, test_size=0.2, random_state=42
    )

    cnn = build_cnn((128, 128, 1))
    print("Extracting CNN features...")
    X_tr_cnn = cnn.predict(X_tr, batch_size=32)
    X_te_cnn = cnn.predict(X_te, batch_size=32)

    X_tr_all = np.hstack((X_tr_cnn, lbp_tr))
    X_te_all = np.hstack((X_te_cnn, lbp_te))

    pca = PCA(n_components=50)
    X_tr_pca = pca.fit_transform(X_tr_all)
    X_te_pca = pca.transform(X_te_all)

    print("Running SMA...")
    mask = sma_feature_selection(X_tr_pca, y_tr)

    X_tr_sel = X_tr_pca[:, mask]
    X_te_sel = X_te_pca[:, mask]

    smote = SMOTE(random_state=42)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_sel, y_tr)

    cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw = dict(enumerate(cw))

    model = build_final_model(X_tr_sel.shape[1])
    early_stop = EarlyStopping(patience=6, restore_best_weights=True)

    history = model.fit(
        X_tr_bal, y_tr_bal,
        epochs=35,
        batch_size=32,
        validation_data=(X_te_sel, y_te),
        class_weight=cw,
        callbacks=[early_stop],
        verbose=1
    )

    y_prob = model.predict(X_te_sel).flatten()

    np.save("roc_ytrue.npy", y_te)
    np.save("roc_sma.npy", y_prob.flatten())

    precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    best_thresh = thresholds[np.argmax(f1[:-1])]
    y_pred = (y_prob > best_thresh).astype(int)

    loss, acc = model.evaluate(X_te_sel, y_te)
    print("Test Loss:", loss)
    print("Test Accuracy:", acc)
    print(classification_report(y_te, y_pred, target_names=categories))

    cm = confusion_matrix(y_te, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    roc = roc_auc_score(y_te, y_prob)
    print("ROC-AUC:", roc)


if __name__ == "__main__":
    main()
