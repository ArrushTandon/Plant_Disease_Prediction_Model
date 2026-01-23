import os
import numpy as np
import cv2
import pickle
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
from imblearn.over_sampling import SMOTE
import tensorflow as tf

# =====================================================
# MIXED PRECISION (ONLY IF GPU EXISTS)
# =====================================================
if tf.config.list_physical_devices('GPU'):
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')


# =====================================================
# LBP FEATURE EXTRACTION
# =====================================================
def compute_lbp(image, P=8, R=1):
    image = (image * 255).astype(np.uint8)
    lbp = local_binary_pattern(image, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, P + 3),
                           range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Invalid image: {image_path}")
    image = cv2.resize(image, target_size)
    image = image / 255.0
    lbp = compute_lbp(image)
    image = np.expand_dims(image, axis=-1)
    return image, lbp


def load_dataset(data_dir, categories):
    images, lbp_features, labels = [], [], []
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        print(f"Loading images from: {category_path}")
        for file in os.listdir(category_path):
            path = os.path.join(category_path, file)
            if not os.path.isfile(path):
                continue
            try:
                img, lbp = load_and_preprocess_image(path)
                images.append(img)
                lbp_features.append(lbp)
                labels.append(label)
            except Exception as e:
                print(f"Skipping {path}: {e}")
    print(f"Loaded {len(images)} images")
    return np.array(images), np.array(lbp_features), np.array(labels)


# =====================================================
# CNN FEATURE EXTRACTOR (UNTRAINED)
# =====================================================
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


# =====================================================
# SLIME MOULD ALGORITHM (SMA) – FEATURE SELECTION
# =====================================================
def sma_feature_selection(
    X, y,
    num_iterations=20,
    population_size=20,
    alpha=0.9
):
    n_samples, n_features = X.shape
    population = np.random.rand(population_size, n_features)

    best_solution = None
    best_fitness = -np.inf
    eps = 1e-8

    for iteration in range(num_iterations):
        fitness = []

        for i in range(population_size):
            mask = population[i] > 0.5
            X_sel = X[:, mask]

            if X_sel.shape[1] == 0:
                score = 0
            else:
                mi = mutual_info_classif(X_sel, y)
                relevance = np.mean(mi)
                sparsity = 1 - (X_sel.shape[1] / n_features)
                score = alpha * relevance + (1 - alpha) * sparsity

            fitness.append(score)

            if score > best_fitness:
                best_fitness = score
                best_solution = population[i].copy()

        fitness = np.array(fitness)
        idx = np.argsort(fitness)[::-1]
        population = population[idx]
        fitness = fitness[idx]

        f_best = fitness[0]
        f_worst = fitness[-1]

        W = np.zeros(population_size)
        for i in range(population_size):
            if i < population_size / 2:
                W[i] = 1 + np.random.rand() * np.log(
                    (f_best - fitness[i]) / (f_best - f_worst + eps) + 1
                )
            else:
                W[i] = 1 - np.random.rand() * np.log(
                    (f_best - fitness[i]) / (f_best - f_worst + eps) + 1
                )

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

        print(f"[SMA] Iteration {iteration+1}/{num_iterations} | Best fitness: {best_fitness:.4f}")

    return best_solution > 0.5


# =====================================================
# FINAL CLASSIFIER
# =====================================================
def build_final_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    data_dir = "data/processed"
    categories = ["healthy", "diseased"]

    images, lbp, labels = load_dataset(data_dir, categories)

    X_tr, X_te, lbp_tr, lbp_te, y_tr, y_te = train_test_split(
        images, lbp, labels, test_size=0.2, random_state=42
    )

    cnn = build_cnn((128, 128, 1))
    X_tr_cnn = cnn.predict(X_tr, batch_size=32)
    X_te_cnn = cnn.predict(X_te, batch_size=32)

    X_tr_all = np.hstack((X_tr_cnn, lbp_tr))
    X_te_all = np.hstack((X_te_cnn, lbp_te))

    pca = PCA(n_components=50)
    X_tr_pca = pca.fit_transform(X_tr_all)
    X_te_pca = pca.transform(X_te_all)

    print("Running Slime Mould Algorithm (SMA)...")
    mask = sma_feature_selection(X_tr_pca, y_tr)

    print(f"Selected features: {mask.sum()} / {len(mask)}")

    X_tr_sel = X_tr_pca[:, mask]
    X_te_sel = X_te_pca[:, mask]

    smote = SMOTE(random_state=42)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_sel, y_tr)

    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_tr), y=y_tr
    )
    class_weights = dict(enumerate(class_weights))

    model = build_final_model(X_tr_sel.shape[1])
    model.fit(
        X_tr_bal, y_tr_bal,
        epochs=20,
        batch_size=32,
        validation_data=(X_te_sel, y_te),
        class_weight=class_weights
    )

    y_prob = model.predict(X_te_sel)
    precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    y_pred = (y_prob > thresholds[np.argmax(f1)]).astype(int)

    print(classification_report(y_te, y_pred, target_names=categories))
    print("ROC-AUC:", roc_auc_score(y_te, y_prob))

    #np.save("models/mask_sma.npy", mask)
    #pickle.dump(pca, open("models/pca_sma.pkl", "wb"))
    #model.save("models/model_cnn_sma.h5")


if __name__ == "__main__":
    main()
