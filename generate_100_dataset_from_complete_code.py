#!/usr/bin/env python3
"""
AURORA-RIS 6G
100-sample Sionna 0.19.2 dataset generator.

Uses the already-verified generate_dataset.py pipeline and preserves:
  BS -> RIS CSI       [5,64,4]
  RIS -> User CSI     [2,5,64]
  BS -> User CSI      [2,5,4]
  Effective channel   [2,5,4]

Each sample gets a different valid user trajectory (position + velocity).
BS, RIS, frequency, bandwidth, number of users, time steps, SNR, CSI error,
and RIS phase resolution remain fixed for this baseline dataset.

Outputs:
  simulation/dataset/output/100_samples/aurora_100_dataset.csv
  simulation/dataset/output/100_samples/aurora_100_metrics.csv
  simulation/dataset/output/100_samples/aurora_100_ris_phases.csv
  simulation/dataset/output/100_samples/aurora_100.h5
  simulation/dataset/output/100_samples/sample_000_3d_scene.png
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "simulation" / "dataset" / "generate_dataset.py"
OUTPUT_DIR = PROJECT_ROOT / "simulation" / "dataset" / "output" / "100_samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 100
BASE_SEED = 20260929

# User sampling region for the simple street-canyon scene.
# Keep z fixed at typical UE height and keep trajectories inside the region.
X_MIN, X_MAX = 8.0, 24.0
Y_MIN, Y_MAX = -8.0, 12.0
Z = 1.5
MIN_USER_SEPARATION = 3.0
VEL_MIN, VEL_MAX = -1.0, 1.0
MAX_ATTEMPTS = 500


def load_pipeline():
    spec = importlib.util.spec_from_file_location("aurora_pipeline", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline: {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_random_trajectory(gd, rng):
    """Generate 2-user, 5-step trajectories that remain in the sampling box."""
    for _ in range(MAX_ATTEMPTS):
        initial = np.zeros((gd.NUM_USERS, 3), dtype=np.float32)
        initial[:, 0] = rng.uniform(X_MIN, X_MAX, gd.NUM_USERS)
        initial[:, 1] = rng.uniform(Y_MIN, Y_MAX, gd.NUM_USERS)
        initial[:, 2] = Z

        distance = np.linalg.norm(initial[0, :2] - initial[1, :2])
        if distance < MIN_USER_SEPARATION:
            continue

        velocities = np.zeros_like(initial)
        velocities[:, 0:2] = rng.uniform(
            VEL_MIN, VEL_MAX, size=(gd.NUM_USERS, 2)
        )

        # Ensure all 5 positions remain in bounds.
        t_end = (gd.TIME_STEPS - 1) * gd.DT
        final = initial + velocities * t_end
        if np.any(final[:, 0] < X_MIN) or np.any(final[:, 0] > X_MAX):
            continue
        if np.any(final[:, 1] < Y_MIN) or np.any(final[:, 1] > Y_MAX):
            continue

        positions = np.zeros(
            (gd.TIME_STEPS, gd.NUM_USERS, 3), dtype=np.float32
        )
        velocities_time = np.zeros_like(positions)

        for t in range(gd.TIME_STEPS):
            positions[t] = initial + velocities * (t * gd.DT)
            velocities_time[t] = velocities

        # Check separation at every time step.
        if any(
            np.linalg.norm(positions[t, 0, :2] - positions[t, 1, :2])
            < MIN_USER_SEPARATION
            for t in range(gd.TIME_STEPS)
        ):
            continue

        return positions, velocities_time

    raise RuntimeError("Could not generate a valid random user trajectory.")


def scalar(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    raise ValueError(f"Expected scalar, got {arr.shape}")


def add_array_to_row(row, prefix, value):
    """Flatten an array into CSV columns, preserving complex real/imag parts."""
    arr = np.asarray(value)
    flat = arr.reshape(-1)

    if np.iscomplexobj(arr):
        for i, v in enumerate(flat):
            row[f"{prefix}_real_{i}"] = float(np.real(v))
            row[f"{prefix}_imag_{i}"] = float(np.imag(v))
    else:
        for i, v in enumerate(flat):
            row[f"{prefix}_{i}"] = v.item()


def append_h5_dataset(h5, name, values):
    arr = np.asarray(values)
    h5.create_dataset(
        name,
        data=arr,
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def main():
    print("=" * 82)
    print("AURORA-RIS 6G | 100-SAMPLE DATASET")
    print("=" * 82)
    print(f"Pipeline : {SOURCE}")
    print(f"Samples  : {N_SAMPLES}")
    print(f"Output   : {OUTPUT_DIR}")
    print("=" * 82)

    gd = load_pipeline()

    # Fixed BS->RIS channel: BS and RIS do not move.
    print("\n[1/4] Computing fixed BS -> RIS CSI once...")
    gd.np.random.seed(BASE_SEED)
    gd.tf.random.set_seed(BASE_SEED)
    fixed_bs_ris = gd.trace_bs_ris()
    if fixed_bs_ris.shape != (
        gd.TIME_STEPS,
        gd.RIS_ELEMENTS,
        gd.BS_ANTENNAS,
    ):
        raise RuntimeError(f"Unexpected BS-RIS shape: {fixed_bs_ris.shape}")

    rows = []
    metric_rows = []
    phase_rows = []

    # Store all arrays for a single HDF5 file.
    h5_arrays = {
        "true_csi_bs_ris": [],
        "true_csi_ris_user": [],
        "true_csi_bs_user": [],
        "csi_bs_ris": [],
        "csi_ris_user": [],
        "csi_bs_user": [],
        "user_position": [],
        "user_velocity": [],
        "continuous_optimal_phase": [],
        "target_phase": [],
        "target_phase_sin_cos": [],
        "optimal_sinr": [],
        "optimal_sum_rate": [],
        "quantized_sum_rate": [],
        "baseline_sum_rate": [],
        "random_sum_rate": [],
        "zero_rx_power_dbm": [],
        "random_rx_power_dbm": [],
        "optimizer_rx_power_dbm": [],
        "zero_sinr": [],
        "random_sinr": [],
        "optimizer_sinr": [],
        "effective_channel_zero": [],
        "effective_channel_random": [],
        "effective_channel_optimizer": [],
        "effective_channel_quantized": [],
    }

    # Fixed geometry verification.
    gd.print_configuration()

    for sample_id in range(N_SAMPLES):
        seed = BASE_SEED + sample_id
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        gd.tf.random.set_seed(seed)

        print("\n" + "-" * 82)
        print(f"SAMPLE {sample_id + 1}/{N_SAMPLES}")
        print("-" * 82)

        # ------------------------------------------------------------
        # Random user trajectory for this sample
        # ------------------------------------------------------------
        positions, velocities = make_random_trajectory(gd, rng)
        gd.INITIAL_USER_POSITIONS = positions[0].copy()
        gd.USER_VELOCITIES = velocities[0].copy()

        # ------------------------------------------------------------
        # Ray-traced channels
        # ------------------------------------------------------------
        scene, tx, ris = gd.create_main_scene()

        if sample_id == 0:
            gd.verify_dimensions(scene, ris)
            gd.trace_actual_ris_support(scene)

        true_csi_bs_user = gd.trace_bs_user(scene, positions)
        true_csi_bs_ris = fixed_bs_ris.copy()
        true_csi_ris_user = gd.trace_ris_user(positions)

        assert true_csi_bs_ris.shape == (
            gd.TIME_STEPS, gd.RIS_ELEMENTS, gd.BS_ANTENNAS
        )
        assert true_csi_ris_user.shape == (
            gd.NUM_USERS, gd.TIME_STEPS, gd.RIS_ELEMENTS
        )
        assert true_csi_bs_user.shape == (
            gd.NUM_USERS, gd.TIME_STEPS, gd.BS_ANTENNAS
        )

        # ------------------------------------------------------------
        # Noise + RIS optimization
        # ------------------------------------------------------------
        zero_phase = np.zeros(gd.RIS_ELEMENTS, dtype=np.float32)
        noise_power_w = gd.calibrate_noise_power(
            true_csi_bs_ris,
            true_csi_ris_user,
            true_csi_bs_user,
            zero_phase,
        )

        random_phase = rng.uniform(
            -np.pi, np.pi, gd.RIS_ELEMENTS
        ).astype(np.float32)

        zero_result = gd.evaluate_cascaded_phase(
            true_csi_bs_ris,
            true_csi_ris_user,
            true_csi_bs_user,
            zero_phase,
            noise_power_w,
        )
        random_result = gd.evaluate_cascaded_phase(
            true_csi_bs_ris,
            true_csi_ris_user,
            true_csi_bs_user,
            random_phase,
            noise_power_w,
        )

        optimized_phase, history = gd.optimize_ris_from_raytraced_csi(
            true_csi_bs_ris,
            true_csi_ris_user,
            true_csi_bs_user,
            noise_power_w,
        )

        optimizer_result = gd.evaluate_cascaded_phase(
            true_csi_bs_ris,
            true_csi_ris_user,
            true_csi_bs_user,
            optimized_phase,
            noise_power_w,
        )

        target_phase = gd.quantize_phase(
            optimized_phase,
            gd.PHASE_BITS,
        )

        quantized_result = gd.evaluate_cascaded_phase(
            true_csi_bs_ris,
            true_csi_ris_user,
            true_csi_bs_user,
            target_phase,
            noise_power_w,
        )

        # Core validity checks for every sample.
        if optimizer_result["h_eff"].shape != (
            gd.NUM_USERS, gd.TIME_STEPS, gd.BS_ANTENNAS
        ):
            raise RuntimeError(
                f"Sample {sample_id}: bad effective channel shape "
                f"{optimizer_result['h_eff'].shape}"
            )
        if not np.all(np.isfinite(np.abs(true_csi_bs_ris))):
            raise RuntimeError(f"Sample {sample_id}: invalid BS-RIS CSI")
        if not np.all(np.isfinite(np.abs(true_csi_ris_user))):
            raise RuntimeError(f"Sample {sample_id}: invalid RIS-user CSI")
        if not np.all(np.isfinite(np.abs(true_csi_bs_user))):
            raise RuntimeError(f"Sample {sample_id}: invalid BS-user CSI")
        if not np.isfinite(optimizer_result["sum_rate_bps"]):
            raise RuntimeError(f"Sample {sample_id}: invalid optimized sum rate")

        # ------------------------------------------------------------
        # Estimated/noisy CSI
        # ------------------------------------------------------------
        csi_bs_ris = gd.add_csi_error(
            true_csi_bs_ris, gd.CSI_ERROR_PERCENT
        )
        csi_ris_user = gd.add_csi_error(
            true_csi_ris_user, gd.CSI_ERROR_PERCENT
        )
        csi_bs_user = gd.add_csi_error(
            true_csi_bs_user, gd.CSI_ERROR_PERCENT
        )

        # ------------------------------------------------------------
        # HDF5 collection
        # ------------------------------------------------------------
        h5_values = {
            "true_csi_bs_ris": true_csi_bs_ris,
            "true_csi_ris_user": true_csi_ris_user,
            "true_csi_bs_user": true_csi_bs_user,
            "csi_bs_ris": csi_bs_ris,
            "csi_ris_user": csi_ris_user,
            "csi_bs_user": csi_bs_user,
            "user_position": positions,
            "user_velocity": velocities,
            "continuous_optimal_phase": optimized_phase.astype(np.float32),
            "target_phase": target_phase.astype(np.float32),
            "target_phase_sin_cos": np.stack(
                [np.sin(target_phase), np.cos(target_phase)], axis=-1
            ).astype(np.float32),
            "optimal_sinr": optimizer_result["sinr"],
            "optimal_sum_rate": np.float32(optimizer_result["sum_rate_bps"]),
            "quantized_sum_rate": np.float32(quantized_result["sum_rate_bps"]),
            "baseline_sum_rate": np.float32(zero_result["sum_rate_bps"]),
            "random_sum_rate": np.float32(random_result["sum_rate_bps"]),
            "zero_rx_power_dbm": zero_result["rx_power_dbm"],
            "random_rx_power_dbm": random_result["rx_power_dbm"],
            "optimizer_rx_power_dbm": optimizer_result["rx_power_dbm"],
            "zero_sinr": zero_result["sinr"],
            "random_sinr": random_result["sinr"],
            "optimizer_sinr": optimizer_result["sinr"],
            "effective_channel_zero": zero_result["h_eff"],
            "effective_channel_random": random_result["h_eff"],
            "effective_channel_optimizer": optimizer_result["h_eff"],
            "effective_channel_quantized": quantized_result["h_eff"],
        }

        for name, value in h5_values.items():
            h5_arrays[name].append(np.asarray(value))

        # ------------------------------------------------------------
        # One-row full CSV sample
        # ------------------------------------------------------------
        row = {
            "sample_id": sample_id,
            "seed": seed,
            "frequency_hz": gd.FREQUENCY_HZ,
            "bandwidth_hz": gd.BANDWIDTH_HZ,
            "snr_db": gd.SNR_DB,
            "csi_error_pct": gd.CSI_ERROR_PERCENT,
            "phase_resolution_bits": gd.PHASE_BITS,
            "bs_antennas": gd.BS_ANTENNAS,
            "ris_elements": gd.RIS_ELEMENTS,
            "num_users": gd.NUM_USERS,
            "time_steps": gd.TIME_STEPS,
            "noise_power_w": float(noise_power_w),
            "baseline_sum_rate_mbps": float(zero_result["sum_rate_bps"] / 1e6),
            "random_sum_rate_mbps": float(random_result["sum_rate_bps"] / 1e6),
            "optimized_sum_rate_mbps": float(optimizer_result["sum_rate_bps"] / 1e6),
            "quantized_sum_rate_mbps": float(quantized_result["sum_rate_bps"] / 1e6),
            "optimized_gain_over_zero_mbps": float(
                (optimizer_result["sum_rate_bps"] - zero_result["sum_rate_bps"]) / 1e6
            ),
        }

        add_array_to_row(row, "user_position", positions)
        add_array_to_row(row, "user_velocity", velocities)
        add_array_to_row(row, "true_csi_bs_ris", true_csi_bs_ris)
        add_array_to_row(row, "true_csi_ris_user", true_csi_ris_user)
        add_array_to_row(row, "true_csi_bs_user", true_csi_bs_user)
        add_array_to_row(row, "csi_bs_ris", csi_bs_ris)
        add_array_to_row(row, "csi_ris_user", csi_ris_user)
        add_array_to_row(row, "csi_bs_user", csi_bs_user)
        add_array_to_row(row, "optimal_phase", optimized_phase)
        add_array_to_row(row, "target_phase", target_phase)
        add_array_to_row(row, "optimal_sinr", optimizer_result["sinr"])
        add_array_to_row(row, "zero_sinr", zero_result["sinr"])
        add_array_to_row(row, "random_sinr", random_result["sinr"])
        add_array_to_row(row, "zero_rx_power_dbm", zero_result["rx_power_dbm"])
        add_array_to_row(row, "random_rx_power_dbm", random_result["rx_power_dbm"])
        add_array_to_row(row, "optimizer_rx_power_dbm", optimizer_result["rx_power_dbm"])
        add_array_to_row(row, "effective_channel_optimizer", optimizer_result["h_eff"])

        rows.append(row)

        metric_rows.append({
            "sample_id": sample_id,
            "seed": seed,
            "snr_db": gd.SNR_DB,
            "csi_error_pct": gd.CSI_ERROR_PERCENT,
            "baseline_sum_rate_mbps": zero_result["sum_rate_bps"] / 1e6,
            "random_sum_rate_mbps": random_result["sum_rate_bps"] / 1e6,
            "optimized_sum_rate_mbps": optimizer_result["sum_rate_bps"] / 1e6,
            "quantized_sum_rate_mbps": quantized_result["sum_rate_bps"] / 1e6,
            "optimized_gain_over_zero_mbps": (
                optimizer_result["sum_rate_bps"] - zero_result["sum_rate_bps"]
            ) / 1e6,
            "user1_sinr": optimizer_result["sinr"][0],
            "user2_sinr": optimizer_result["sinr"][1],
            "user1_rx_power_dbm": optimizer_result["rx_power_dbm"][0],
            "user2_rx_power_dbm": optimizer_result["rx_power_dbm"][1],
            "optimizer_iterations": len(history),
        })

        phase_row = {"sample_id": sample_id}
        for i, phase in enumerate(target_phase):
            phase_row[f"phase_{i}"] = float(phase)
        phase_rows.append(phase_row)

        print(
            f"Sample {sample_id + 1:03d}/{N_SAMPLES}: "
            f"users={positions[0,:,0:2].round(2).tolist()} | "
            f"optimized={optimizer_result['sum_rate_bps']/1e6:.4f} Mbps | "
            f"gain={(optimizer_result['sum_rate_bps']-zero_result['sum_rate_bps'])/1e6:.4f} Mbps"
        )

        if sample_id == 0:
            # Generate only one visualization; doing this 100 times is unnecessary.
            try:
                scene_path = OUTPUT_DIR / "sample_000_3d_scene.png"
                gd.create_3d_visualization(positions)
                default_scene = gd.OUTPUT_DIR / "single_sample_3d_scene.png"
                if default_scene.exists():
                    default_scene.replace(scene_path)
                print(f"3D scene: {scene_path}")
            except Exception as exc:
                print(f"WARNING: 3D visualization skipped: {exc}")

    # ------------------------------------------------------------
    # Write full HDF5 dataset
    # ------------------------------------------------------------
    h5_path = OUTPUT_DIR / "aurora_100.h5"
    if h5_path.exists():
        h5_path.unlink()

    print("\n[2/4] Writing HDF5 dataset...")
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["project"] = "AURORA-RIS 6G"
        h5.attrs["dataset_stage"] = "100-sample-baseline"
        h5.attrs["num_samples"] = N_SAMPLES
        h5.attrs["sionna_version"] = str(gd.sionna.__version__)
        h5.attrs["frequency_hz"] = gd.FREQUENCY_HZ
        h5.attrs["bandwidth_hz"] = gd.BANDWIDTH_HZ
        h5.attrs["bs_antennas"] = gd.BS_ANTENNAS
        h5.attrs["ris_elements"] = gd.RIS_ELEMENTS
        h5.attrs["num_users"] = gd.NUM_USERS
        h5.attrs["time_steps"] = gd.TIME_STEPS
        h5.attrs["snr_db"] = gd.SNR_DB
        h5.attrs["csi_error_pct"] = gd.CSI_ERROR_PERCENT
        h5.attrs["phase_bits"] = gd.PHASE_BITS

        for name, values in h5_arrays.items():
            append_h5_dataset(h5, name, values)

    # ------------------------------------------------------------
    # CSV writers
    # ------------------------------------------------------------
    print("[3/4] Writing CSV files...")

    full_csv = OUTPUT_DIR / "aurora_100_dataset.csv"
    with full_csv.open("w", newline="") as fp:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics_csv = OUTPUT_DIR / "aurora_100_metrics.csv"
    with metrics_csv.open("w", newline="") as fp:
        fieldnames = list(metric_rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    phases_csv = OUTPUT_DIR / "aurora_100_ris_phases.csv"
    with phases_csv.open("w", newline="") as fp:
        fieldnames = list(phase_rows[0].keys())
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(phase_rows)

    print("[4/4] Final validation...")
    with h5py.File(h5_path, "r") as h5:
        assert h5["true_csi_bs_ris"].shape == (
            N_SAMPLES, gd.TIME_STEPS, gd.RIS_ELEMENTS, gd.BS_ANTENNAS
        )
        assert h5["true_csi_ris_user"].shape == (
            N_SAMPLES, gd.NUM_USERS, gd.TIME_STEPS, gd.RIS_ELEMENTS
        )
        assert h5["true_csi_bs_user"].shape == (
            N_SAMPLES, gd.NUM_USERS, gd.TIME_STEPS, gd.BS_ANTENNAS
        )
        assert h5["effective_channel_optimizer"].shape == (
            N_SAMPLES, gd.NUM_USERS, gd.TIME_STEPS, gd.BS_ANTENNAS
        )

    print("\n" + "=" * 82)
    print("100-SAMPLE DATASET COMPLETE")
    print("=" * 82)
    print(f"Samples             : {N_SAMPLES}")
    print(f"Full CSV            : {full_csv}")
    print(f"Metrics CSV         : {metrics_csv}")
    print(f"RIS phases CSV      : {phases_csv}")
    print(f"HDF5                : {h5_path}")
    print(f"3D scene            : {OUTPUT_DIR / 'sample_000_3d_scene.png'}")
    print("\nVerified shapes:")
    print(f"  BS-RIS             : ({N_SAMPLES}, 5, 64, 4)")
    print(f"  RIS-User           : ({N_SAMPLES}, 2, 5, 64)")
    print(f"  BS-User            : ({N_SAMPLES}, 2, 5, 4)")
    print(f"  Effective channel  : ({N_SAMPLES}, 2, 5, 4)")
    print("=" * 82)


if __name__ == "__main__":
    main()
