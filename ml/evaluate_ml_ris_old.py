import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS / CONFIG
# ============================================================

H5_PATH = Path(
    "simulation/dataset/output/1000_samples/aurora_1000.h5"
)

MODEL_PATH = Path(
    "ml/models/best_ris_phase_model.keras"
)

NORM_PATH = Path(
    "ml/models/normalization.npz"
)

OUTPUT_DIR = Path(
    "ml/evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42

NUM_SAMPLES = 1000
NUM_USERS = 2
TIME_STEPS = 5
RIS_ELEMENTS = 64
BS_ANTENNAS = 4

BANDWIDTH_HZ = 20e6
TX_POWER_DBM = 23.0
SNR_DB = 10.0

EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def dbm_to_watt(dbm):
    return 10.0 ** ((dbm - 30.0) / 10.0)


def watt_to_dbm(watt):
    return 10.0 * np.log10(
        max(float(watt), EPS)
    ) + 30.0


def complex_to_realimag(x):
    """
    Same feature representation used during training.
    """
    x = np.asarray(x)

    return np.concatenate(
        [
            x.real.reshape(x.shape[0], -1),
            x.imag.reshape(x.shape[0], -1),
        ],
        axis=1
    ).astype(np.float32)


def build_features(
    bs_ris,
    ris_user,
    bs_user
):
    x1 = complex_to_realimag(bs_ris)
    x2 = complex_to_realimag(ris_user)
    x3 = complex_to_realimag(bs_user)

    return np.concatenate(
        [x1, x2, x3],
        axis=1
    ).astype(np.float32)


def wrap_phase(x):
    return (
        (x + np.pi) % (2.0 * np.pi)
    ) - np.pi


def quantize_phase(
    phase,
    bits=2
):
    levels = 2 ** bits

    step = (
        2.0 * np.pi
        / levels
    )

    q = np.round(
        wrap_phase(phase) / step
    ) * step

    return wrap_phase(q).astype(
        np.float32
    )


# ============================================================
# EFFECTIVE RIS CHANNEL
# ============================================================

def effective_channel(
    h_bs_ris,
    h_ris_user,
    h_bs_user,
    phase
):
    """
    h_bs_ris  : [T, R, M]
    h_ris_user: [U, T, R]
    h_bs_user : [U, T, M]
    phase     : [R]

    Returns:
        [U, T, M]
    """

    h_br = np.asarray(
        h_bs_ris,
        dtype=np.complex64
    )

    h_ru = np.asarray(
        h_ris_user,
        dtype=np.complex64
    )

    h_bu = np.asarray(
        h_bs_user,
        dtype=np.complex64
    )

    phase_factor = np.exp(
        1j * phase
    ).astype(np.complex64)

    cascaded = np.sum(
        h_ru[..., None]
        * h_br[None, ...]
        * phase_factor[None, None, :, None],
        axis=2
    )

    return (
        h_bu + cascaded
    ).astype(np.complex64)


# ============================================================
# ZF METRICS
# Same mathematical evaluation used by the dataset pipeline
# ============================================================

def zf_metrics(
    h_user_bs,
    noise_power_w
):
    """
    h_user_bs: [U, M]
    """

    h = np.asarray(
        h_user_bs,
        dtype=np.complex64
    )

    hh = (
        h
        @ h.conj().T
    )

    reg = (
        1e-9
        * np.eye(
            h.shape[0],
            dtype=np.complex64
        )
    )

    inv = np.linalg.inv(
        hh + reg
    )

    w = (
        h.conj().T
        @ inv
    )

    w_norm = np.sqrt(
        np.sum(
            np.abs(w) ** 2,
            axis=0,
            keepdims=True
        )
        + EPS
    )

    w = w / w_norm

    tx_power = dbm_to_watt(
        TX_POWER_DBM
    )

    power_per_user = (
        tx_power
        / h.shape[0]
    )

    effective = h @ w

    signal = np.abs(
        np.diag(effective)
    ) ** 2

    total = np.abs(
        effective
    ) ** 2

    interference = (
        np.sum(
            total,
            axis=1
        )
        - signal
    )

    signal_power = (
        power_per_user
        * signal
    )

    interference_power = (
        power_per_user
        * np.maximum(
            interference,
            0.0
        )
    )

    sinr = (
        signal_power
        / (
            interference_power
            + noise_power_w
            + EPS
        )
    )

    rate = (
        BANDWIDTH_HZ
        * np.log2(
            1.0 + sinr
        )
    )

    sum_rate = np.sum(
        rate
    )

    return (
        signal_power.astype(np.float32),
        sinr.astype(np.float32),
        float(sum_rate)
    )


# ============================================================
# NOISE CALIBRATION
# Same calibration logic as dataset generator
# ============================================================

def calibrate_noise_power(
    h_bs_ris,
    h_ris_user,
    h_bs_user
):
    zero_phase = np.zeros(
        RIS_ELEMENTS,
        dtype=np.float32
    )

    h_eff = effective_channel(
        h_bs_ris,
        h_ris_user,
        h_bs_user,
        zero_phase
    )

    h0 = h_eff[:, 0, :]

    reference_rx_power, _, _ = zf_metrics(
        h0,
        1e-30
    )

    reference_power = float(
        np.mean(
            reference_rx_power
        )
    )

    target_linear_snr = (
        10.0 ** (
            SNR_DB / 10.0
        )
    )

    noise_power = (
        reference_power
        / target_linear_snr
    )

    return max(
        noise_power,
        1e-30
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 75)
print("AURORA-RIS 6G | ML RIS PERFORMANCE EVALUATION")
print("=" * 75)

# ------------------------------------------------------------
# Load H5
# ------------------------------------------------------------

print("\nLoading H5 dataset...")

with h5py.File(
    H5_PATH,
    "r"
) as f:

    true_bs_ris = f[
        "true_csi_bs_ris"
    ][:]

    true_ris_user = f[
        "true_csi_ris_user"
    ][:]

    true_bs_user = f[
        "true_csi_bs_user"
    ][:]

    target_phase = f[
        "target_phase"
    ][:]

    target_phase_sin_cos = f[
        "target_phase_sin_cos"
    ][:]

    optimizer_sum_rate = f[
        "optimal_sum_rate"
    ][:]

    baseline_sum_rate = f[
        "baseline_sum_rate"
    ][:]

    random_sum_rate = f[
        "random_sum_rate"
    ][:]

    quantized_sum_rate = f[
        "quantized_sum_rate"
    ][:]

    optimizer_sinr = f[
        "optimizer_sinr"
    ][:]

    zero_sinr = f[
        "zero_sinr"
    ][:]

    random_sinr = f[
        "random_sinr"
    ][:]

print("BS-RIS CSI :", true_bs_ris.shape)
print("RIS-User   :", true_ris_user.shape)
print("BS-User    :", true_bs_user.shape)
print("Target     :", target_phase.shape)


# ------------------------------------------------------------
# Reconstruct EXACT 80/10/10 split
# ------------------------------------------------------------

print("\nReconstructing 80/10/10 split...")

indices = np.arange(
    NUM_SAMPLES
)

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED
)

print(
    f"Train      : {len(train_idx)} samples"
)

print(
    f"Validation : {len(val_idx)} samples"
)

print(
    f"Test       : {len(test_idx)} samples"
)


# ------------------------------------------------------------
# Build ML input exactly like training
# ------------------------------------------------------------

print("\nBuilding test features...")

X_all = build_features(
    true_bs_ris,
    true_ris_user,
    true_bs_user
)

X_test = X_all[
    test_idx
]


# ------------------------------------------------------------
# Load normalization
# ------------------------------------------------------------

norm = np.load(
    NORM_PATH
)

mean = norm[
    "mean"
]

std = norm[
    "std"
]

X_test_scaled = (
    X_test - mean
) / std


# ------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------

print("\nLoading trained ML model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ------------------------------------------------------------
# ML prediction
# ------------------------------------------------------------

print("\nGenerating ML RIS phase predictions...")

pred = model.predict(
    X_test_scaled,
    verbose=1
)

pred = pred.reshape(
    -1,
    RIS_ELEMENTS,
    2
)

# IMPORTANT:
# target = [sin(phase), cos(phase)]
# Therefore:
# phase = atan2(sin, cos)

ml_phase = np.arctan2(
    pred[:, :, 0],
    pred[:, :, 1]
).astype(
    np.float32
)

true_phase = target_phase[
    test_idx
].astype(
    np.float32
)


# ------------------------------------------------------------
# Circular phase error
# ------------------------------------------------------------

phase_error = np.angle(
    np.exp(
        1j * (
            ml_phase
            - true_phase
        )
    )
)

phase_mae = float(
    np.mean(
        np.abs(
            phase_error
        )
    )
)

phase_rmse = float(
    np.sqrt(
        np.mean(
            phase_error ** 2
        )
    )
)

print(
    f"\nML Phase MAE  : {phase_mae:.6f} rad"
)

print(
    f"ML Phase RMSE : {phase_rmse:.6f} rad"
)


# ============================================================
# EVALUATE EVERY TEST SAMPLE
# ============================================================

results = []

ml_sum_rates = []
ml_sinrs = []
ml_rx_power = []

ml_quantized_sum_rates = []

for local_i, sample_idx in enumerate(
    test_idx
):

    h_br = true_bs_ris[
        sample_idx
    ]

    h_ru = true_ris_user[
        sample_idx
    ]

    h_bu = true_bs_user[
        sample_idx
    ]

    noise_power = calibrate_noise_power(
        h_br,
        h_ru,
        h_bu
    )

    phase = ml_phase[
        local_i
    ]

    # ML continuous phase
    h_ml = effective_channel(
        h_br,
        h_ru,
        h_bu,
        phase
    )

    rx_w, sinr, sum_rate = zf_metrics(
        h_ml[:, 0, :],
        noise_power
    )

    rx_dbm = np.array(
        [
            watt_to_dbm(x)
            for x in rx_w
        ],
        dtype=np.float32
    )

    # ML 2-bit quantized phase
    ml_q_phase = quantize_phase(
        phase,
        bits=2
    )

    h_ml_q = effective_channel(
        h_br,
        h_ru,
        h_bu,
        ml_q_phase
    )

    _, _, ml_q_sum_rate = zf_metrics(
        h_ml_q[:, 0, :],
        noise_power
    )

    ml_sum_rates.append(
        sum_rate
    )

    ml_sinrs.append(
        sinr
    )

    ml_rx_power.append(
        rx_dbm
    )

    results.append(
        {
            "test_order": local_i,
            "sample_id": int(sample_idx),

            "baseline_sum_rate_mbps":
                float(
                    baseline_sum_rate[
                        sample_idx
                    ] / 1e6
                ),

            "random_sum_rate_mbps":
                float(
                    random_sum_rate[
                        sample_idx
                    ] / 1e6
                ),

            "optimizer_sum_rate_mbps":
                float(
                    optimizer_sum_rate[
                        sample_idx
                    ] / 1e6
                ),

            "quantized_sum_rate_mbps":
                float(
                    quantized_sum_rate[
                        sample_idx
                    ] / 1e6
                ),

            "ml_sum_rate_mbps":
                float(
                    sum_rate / 1e6
                ),

            "ml_quantized_sum_rate_mbps":
                float(
                    ml_q_sum_rate / 1e6
                ),

            "ml_gain_over_zero_mbps":
                float(
                    (
                        sum_rate
                        - baseline_sum_rate[
                            sample_idx
                        ]
                    ) / 1e6
                ),

            "phase_mae_rad":
                float(
                    np.mean(
                        np.abs(
                            phase_error[
                                local_i
                            ]
                        )
                    )
                )
        }
    )


ml_sum_rates = np.asarray(
    ml_sum_rates
)

ml_sinrs = np.asarray(
    ml_sinrs
)

ml_rx_power = np.asarray(
    ml_rx_power
)


# ============================================================
# SAVE RESULTS CSV
# ============================================================

results_df = pd.DataFrame(
    results
)

results_csv = (
    OUTPUT_DIR
    / "ml_vs_baselines_test_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "method": [
            "Zero",
            "Random",
            "Optimizer",
            "Quantized Optimizer",
            "ML Continuous",
            "ML Quantized"
        ],

        "mean_sum_rate_mbps": [
            np.mean(
                baseline_sum_rate[
                    test_idx
                ]
            ) / 1e6,

            np.mean(
                random_sum_rate[
                    test_idx
                ]
            ) / 1e6,

            np.mean(
                optimizer_sum_rate[
                    test_idx
                ]
            ) / 1e6,

            np.mean(
                quantized_sum_rate[
                    test_idx
                ]
            ) / 1e6,

            np.mean(
                ml_sum_rates
            ) / 1e6,

            np.mean(
                [
                    r[
                        "ml_quantized_sum_rate_mbps"
                    ]
                    for r in results
                ]
            )
        ]
    }
)

summary_csv = (
    OUTPUT_DIR
    / "ml_vs_baselines_summary.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL ML EVALUATION")
print("=" * 75)

print(
    f"Test samples              : {len(test_idx)}"
)

print(
    f"Phase MAE                 : {phase_mae:.6f} rad"
)

print(
    f"Phase RMSE                : {phase_rmse:.6f} rad"
)

for _, row in summary.iterrows():

    print(
        f"{row['method']:22s}: "
        f"{row['mean_sum_rate_mbps']:.4f} Mbps"
    )


# ============================================================
# GRAPH 1: SUM RATE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    summary["method"],
    summary["mean_sum_rate_mbps"]
)

plt.ylabel(
    "Mean Sum Rate (Mbps)"
)

plt.xlabel(
    "Method"
)

plt.title(
    "RIS Sum Rate Comparison - 100 Test Samples"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "01_sum_rate_comparison.png",
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 2: ML vs OPTIMIZER
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    optimizer_sum_rate[
        test_idx
    ] / 1e6,

    ml_sum_rates / 1e6,

    alpha=0.7
)

min_v = min(
    np.min(
        optimizer_sum_rate[
            test_idx
        ] / 1e6
    ),

    np.min(
        ml_sum_rates / 1e6
    )
)

max_v = max(
    np.max(
        optimizer_sum_rate[
            test_idx
        ] / 1e6
    ),

    np.max(
        ml_sum_rates / 1e6
    )
)

plt.plot(
    [min_v, max_v],
    [min_v, max_v]
)

plt.xlabel(
    "Optimizer Sum Rate (Mbps)"
)

plt.ylabel(
    "ML Sum Rate (Mbps)"
)

plt.title(
    "ML vs Optimizer Sum Rate"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "02_ml_vs_optimizer.png",
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 3: SINR
# ============================================================

mean_zero_sinr = np.mean(
    zero_sinr[
        test_idx
    ],
    axis=0
)

mean_random_sinr = np.mean(
    random_sinr[
        test_idx
    ],
    axis=0
)

mean_optimizer_sinr = np.mean(
    optimizer_sinr[
        test_idx
    ],
    axis=0
)

mean_ml_sinr = np.mean(
    ml_sinrs,
    axis=0
)

x = np.arange(
    NUM_USERS
)

width = 0.2

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - 1.5 * width,
    mean_zero_sinr,
    width,
    label="Zero"
)

plt.bar(
    x - 0.5 * width,
    mean_random_sinr,
    width,
    label="Random"
)

plt.bar(
    x + 0.5 * width,
    mean_optimizer_sinr,
    width,
    label="Optimizer"
)

plt.bar(
    x + 1.5 * width,
    mean_ml_sinr,
    width,
    label="ML"
)

plt.xlabel(
    "User"
)

plt.ylabel(
    "Mean SINR"
)

plt.title(
    "SINR Comparison"
)

plt.xticks(
    x,
    ["User 1", "User 2"]
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "03_sinr_comparison.png",
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 4: PHASE ERROR
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.hist(
    np.degrees(
        phase_error
    ).flatten(),
    bins=40
)

plt.xlabel(
    "Circular Phase Error (degrees)"
)

plt.ylabel(
    "Count"
)

plt.title(
    "ML RIS Phase Prediction Error"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "04_phase_error_distribution.png",
    dpi=200
)

plt.close()


# ============================================================
# GRAPH 5: TRUE VS ML PHASE FOR FIRST TEST SAMPLE
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    np.degrees(
        true_phase[0]
    ),
    label="True / Target"
)

plt.plot(
    np.degrees(
        ml_phase[0]
    ),
    label="ML Predicted"
)

plt.xlabel(
    "RIS Element"
)

plt.ylabel(
    "Phase (degrees)"
)

plt.title(
    "True vs ML Predicted RIS Phase - First Test Sample"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "05_true_vs_ml_phase.png",
    dpi=200
)

plt.close()


# ============================================================
# 3D RIS PHASE VISUALIZATION
# ============================================================

phase_surface = ml_phase[0].reshape(
    8,
    8
)

x_grid, y_grid = np.meshgrid(
    np.arange(8),
    np.arange(8)
)

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot_surface(
    x_grid,
    y_grid,
    np.degrees(
        phase_surface
    )
)

ax.set_xlabel(
    "RIS Column"
)

ax.set_ylabel(
    "RIS Row"
)

ax.set_zlabel(
    "Predicted Phase (degrees)"
)

ax.set_title(
    "ML Predicted 8×8 RIS Phase Surface"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "06_ml_ris_phase_3d.png",
    dpi=200
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

np.savez(
    OUTPUT_DIR
    / "ml_test_predictions.npz",

    sample_indices=test_idx,

    predicted_phase=ml_phase,

    true_phase=true_phase,

    phase_error=phase_error,

    ml_sum_rate_bps=ml_sum_rates,

    ml_sinr=ml_sinrs,

    ml_rx_power_dbm=ml_rx_power
)


print("\nFiles generated:")
print(
    "  ",
    results_csv
)

print(
    "  ",
    summary_csv
)

print(
    "  ml_test_predictions.npz"
)

print(
    "  01_sum_rate_comparison.png"
)

print(
    "  02_ml_vs_optimizer.png"
)

print(
    "  03_sinr_comparison.png"
)

print(
    "  04_phase_error_distribution.png"
)

print(
    "  05_true_vs_ml_phase.png"
)

print(
    "  06_ml_ris_phase_3d.png"
)

print("\nEVALUATION COMPLETE.")
