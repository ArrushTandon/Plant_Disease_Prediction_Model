import os
import numpy as np
import cv2
import pickle
import time
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Enable mixed precision only if GPU exists
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
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Invalid image: {image_path}")
    image = cv2.resize(image, target_size)
    image = image / 255.0
    lbp_features = compute_lbp(image)
    image = np.expand_dims(image, axis=-1)
    return image, lbp_features


def load_dataset(data_dir, categories):
    images, lbp_features, labels = [], [], []
    for label, category in enumerate(categories):
        path = os.path.join(data_dir, category)
        print(f"Loading: {path}")
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue
            try:
                img, lbp = load_and_preprocess_image(file_path)
                images.append(img)
                lbp_features.append(lbp)
                labels.append(label)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
    print(f"Total images loaded: {len(images)}")
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


# -------------------- DBO FEATURE SELECTION --------------------
def dbo_feature_selection(X, y, iterations=20, population_size=20, alpha=0.9):
    n_features = X.shape[1]
    population = np.random.randint(0, 2, (population_size, n_features))
    best_solution = None
    best_fitness = -np.inf

    for it in range(iterations):
        fitness = []

        for i in range(population_size):
            mask = population[i].astype(bool)
            if mask.sum() == 0:
                score = 0
            else:
                X_sel = X[:, mask]
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

        # DBO behaviors
        for i in range(population_size // 2, population_size):
            r = np.random.rand()
            if r < 0.4:      # Ball rolling
                j = np.random.randint(population_size // 2)
                population[i] = population[j] ^ np.random.randint(0, 2, n_features)
            elif r < 0.7:    # Foraging
                population[i] |= population[0]
            else:            # Stealing
                j = np.random.randint(population_size // 2)
                population[i] = population[j].copy()

            if np.random.rand() < 0.1:
                k = np.random.randint(n_features)
                population[i, k] ^= 1

        print(f"[DBO] Iteration {it+1}/{iterations} | Best fitness: {best_fitness:.4f}")

    return best_solution.astype(bool)


# -------------------- CLASSIFIER --------------------
def build_classifier(input_dim):
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
    X_tr_cnn = cnn.predict(X_tr, batch_size=32)
    X_te_cnn = cnn.predict(X_te, batch_size=32)

    X_tr_all = np.hstack((X_tr_cnn, lbp_tr))
    X_te_all = np.hstack((X_te_cnn, lbp_te))

    pca = PCA(n_components=50)
    X_tr_pca = pca.fit_transform(X_tr_all)
    X_te_pca = pca.transform(X_te_all)

    print("Running DBO feature selection...")
    mask = dbo_feature_selection(X_tr_pca, y_tr)

    X_tr_sel = X_tr_pca[:, mask]
    X_te_sel = X_te_pca[:, mask]

    smote = SMOTE(random_state=42)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_tr_sel, y_tr)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    class_weights = dict(enumerate(class_weights))

    model = build_classifier(X_tr_sel.shape[1])
    model.fit(
        X_tr_bal, y_tr_bal,
        epochs=20,
        batch_size=32,
        validation_data=(X_te_sel, y_te),
        class_weight=class_weights
    )

    #model.save("models/model_cnn_dbo.h5")
    #np.save("models/mask_dbo.npy", mask)
    #pickle.dump(pca, open("models/pca_dbo.pkl", "wb"))

    y_prob = model.predict(X_te_sel)
    precision, recall, thresholds = precision_recall_curve(y_te, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    y_pred = (y_prob > thresholds[np.argmax(f1)]).astype(int)

    print(classification_report(y_te, y_pred, target_names=categories))
    print("ROC-AUC:", roc_auc_score(y_te, y_prob))


if __name__ == "__main__":
    main()
