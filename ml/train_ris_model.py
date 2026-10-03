import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

H5_PATH = Path("simulation/dataset/output/1000_samples/aurora_1000.h5")
MODEL_DIR = Path("ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 70)
print("AURORA-RIS 6G | ML RIS PHASE TRAINING")
print("=" * 70)

with h5py.File(H5_PATH, "r") as f:
    bs_ris = f["true_csi_bs_ris"][:]
    ris_user = f["true_csi_ris_user"][:]
    bs_user = f["true_csi_bs_user"][:]
    target = f["target_phase_sin_cos"][:]

print("BS-RIS CSI      :", bs_ris.shape)
print("RIS-User CSI    :", ris_user.shape)
print("BS-User CSI     :", bs_user.shape)
print("Target phases   :", target.shape)

def complex_to_realimag(x):
    x = np.asarray(x)
    return np.concatenate(
        [x.real.reshape(x.shape[0], -1),
         x.imag.reshape(x.shape[0], -1)],
        axis=1
    ).astype(np.float32)

X1 = complex_to_realimag(bs_ris)
X2 = complex_to_realimag(ris_user)
X3 = complex_to_realimag(bs_user)

X = np.concatenate([X1, X2, X3], axis=1)

# Flatten sin/cos target: 64 phases × 2 values
y = target.reshape(target.shape[0], -1).astype(np.float32)

print("ML input shape  :", X.shape)
print("ML target shape :", y.shape)

# Train / validation / test = 80 / 10 / 10
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED
)

# Normalize using training data only
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std < 1e-8] = 1.0

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

np.savez(
    MODEL_DIR / "normalization.npz",
    mean=mean,
    std=std
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(256, activation="relu"),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dense(128, activation="linear")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=["mae"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_DIR / "best_ris_phase_model.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

test_loss, test_mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Test MSE : {test_loss:.6f}")
print(f"Test MAE : {test_mae:.6f}")

# Convert predicted sin/cos back to 64 phase values
pred = model.predict(X_test, verbose=0)
pred = pred.reshape(-1, 64, 2)

pred_phase = np.arctan2(pred[:, :, 1], pred[:, :, 0])

true = y_test.reshape(-1, 64, 2)
true_phase = np.arctan2(true[:, :, 1], true[:, :, 0])

# Circular phase error
phase_error = np.angle(
    np.exp(1j * (pred_phase - true_phase))
)

mae_phase = np.mean(np.abs(phase_error))
rmse_phase = np.sqrt(np.mean(phase_error ** 2))

print(f"Phase MAE  : {mae_phase:.6f} rad")
print(f"Phase RMSE : {rmse_phase:.6f} rad")

np.savez(
    MODEL_DIR / "test_predictions.npz",
    predicted_phase=pred_phase,
    true_phase=true_phase
)

print("\nSaved:")
print("  ml/models/best_ris_phase_model.keras")
print("  ml/models/normalization.npz")
print("  ml/models/test_predictions.npz")
