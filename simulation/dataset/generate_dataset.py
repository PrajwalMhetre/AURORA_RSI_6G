#!/usr/bin/env python3
"""
AURORA-RIS 6G
One-sample verification pipeline for Sionna 0.19.2.

IMPORTANT:
- Generates ONE sample only.
- Uses Sionna RT ray tracing for:
    1) BS -> RIS element channels
    2) RIS element -> User channels
    3) BS -> User direct channels
- BS has exactly 4 antennas.
- RIS has exactly 8x8 = 64 elements.
- RIS optimization is performed on the ray-traced cascaded CSI:
      H_eff[u,m] = sum_n H_ru[u,n] * exp(j*phi[n]) * H_br[n,m]
  This makes the optimization objective explicitly phase-dependent and
  avoids the previous "all-zero phase" failure.
- A real Sionna RIS object is also traced with ris=True for verification.
- Produces train.h5 / validation.h5 / test.h5 containing ONE identical sample.
- Does NOT generate a large dataset.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sionna

from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    RIS,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "simulation" / "dataset" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SYSTEM CONFIGURATION
# ============================================================

FREQUENCY_HZ = 28e9
BANDWIDTH_HZ = 20e6

TX_POWER_DBM = 23.0
SNR_DB = 10.0
CSI_ERROR_PERCENT = 10.0
PHASE_BITS = 2

BS_ROWS = 2
BS_COLS = 2
BS_ANTENNAS = BS_ROWS * BS_COLS

RIS_ROWS = 8
RIS_COLS = 8
RIS_ELEMENTS = RIS_ROWS * RIS_COLS

NUM_USERS = 2
TIME_STEPS = 5
DT = 0.1

# Gradient optimizer
OPT_RESTARTS = 3
OPT_ITERS = 60
LEARNING_RATE = 0.08

EPS = 1e-12


# ============================================================
# FIXED GEOMETRY
# ============================================================

BS_POSITION = np.array(
    [-18.0, -10.0, 12.0],
    dtype=np.float32,
)

RIS_POSITION = np.array(
    [0.0, -2.0, 6.0],
    dtype=np.float32,
)

INITIAL_USER_POSITIONS = np.array(
    [
        [14.0, -4.0, 1.5],
        [18.0,  7.0, 1.5],
    ],
    dtype=np.float32,
)

USER_VELOCITIES = np.array(
    [
        [0.8,  0.25, 0.0],
        [-0.4, -0.15, 0.0],
    ],
    dtype=np.float32,
)


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def dbm_to_watt(dbm: float) -> float:
    return float(10.0 ** ((float(dbm) - 30.0) / 10.0))


def watt_to_dbm(power_w: float) -> float:
    return 10.0 * math.log10(max(float(power_w), EPS)) + 30.0


def wrap_phase_np(phase: np.ndarray) -> np.ndarray:
    return ((phase + np.pi) % (2.0 * np.pi)) - np.pi


def wrap_phase_tf(phase: tf.Tensor) -> tf.Tensor:
    return tf.atan2(tf.sin(phase), tf.cos(phase))


def make_user_trajectories():
    positions = np.zeros(
        (TIME_STEPS, NUM_USERS, 3),
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)

    for t in range(TIME_STEPS):
        positions[t] = (
            INITIAL_USER_POSITIONS
            + USER_VELOCITIES * (t * DT)
        )
        velocities[t] = USER_VELOCITIES

    return positions, velocities


# ============================================================
# CONFIGURATION PRINT
# ============================================================

def print_configuration():
    print("=" * 82)
    print("AURORA-RIS 6G | ONE-SAMPLE VERIFICATION")
    print("=" * 82)
    print(f"Python       : 3.11+")
    print(f"TensorFlow   : {tf.__version__}")
    print(f"Sionna       : {sionna.__version__}")
    print(f"Frequency    : {FREQUENCY_HZ / 1e9:.1f} GHz")
    print(f"Bandwidth    : {BANDWIDTH_HZ / 1e6:.1f} MHz")
    print(f"BS           : {BS_ROWS}x{BS_COLS} = {BS_ANTENNAS} antennas")
    print(f"RIS          : {RIS_ROWS}x{RIS_COLS} = {RIS_ELEMENTS} elements")
    print(f"Users        : {NUM_USERS}")
    print(f"Time steps   : {TIME_STEPS}")
    print(f"SNR          : {SNR_DB:.1f} dB")
    print(f"CSI error    : {CSI_ERROR_PERCENT:.1f}%")
    print(f"Phase bits   : {PHASE_BITS}")
    print("=" * 82)


# ============================================================
# GENERIC SIONNA CIR EXTRACTION
# ============================================================

def require_shape_rank(x: np.ndarray, expected_rank: int, name: str):
    if x.ndim != expected_rank:
        raise RuntimeError(
            f"{name}: expected rank {expected_rank}, got "
            f"shape {x.shape}"
        )


def squeeze_singleton_axes(
    x: np.ndarray,
    axes: tuple[int, ...],
    name: str,
) -> np.ndarray:
    """
    Squeeze only axes that are actually singleton.

    This fixes the previous BS-RIS bug where axis 2 was assumed to be
    singleton although it is the 64-element RIS receive dimension.
    """
    y = x
    for axis in sorted(axes, reverse=True):
        if axis >= y.ndim:
            raise RuntimeError(
                f"{name}: axis {axis} does not exist in shape {y.shape}"
            )
        if y.shape[axis] != 1:
            raise RuntimeError(
                f"{name}: cannot squeeze axis {axis} in shape {y.shape}; "
                f"dimension is {y.shape[axis]}, not 1."
            )
        y = np.squeeze(y, axis=axis)
    return y


def coherent_sum_paths(x: np.ndarray, path_axis: int = -1):
    return np.sum(x, axis=path_axis)


# ============================================================
# SIONNA SCENE
# ============================================================

def create_main_scene():
    scene = load_scene(sionna.rt.scene.simple_street_canyon)

    scene.frequency = FREQUENCY_HZ
    scene.bandwidth = BANDWIDTH_HZ

    # EXACTLY 4 BS ANTENNAS
    scene.tx_array = PlanarArray(
        num_rows=BS_ROWS,
        num_cols=BS_COLS,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    # ONE ANTENNA PER USER
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    tx = Transmitter(
        name="BS",
        position=BS_POSITION,
        look_at=RIS_POSITION,
        power_dbm=TX_POWER_DBM,
    )
    scene.add(tx)

    ris = RIS(
        name="RIS",
        position=RIS_POSITION,
        num_rows=RIS_ROWS,
        num_cols=RIS_COLS,
        num_modes=1,
        look_at=(
            BS_POSITION
            + np.mean(INITIAL_USER_POSITIONS, axis=0)
        ) / 2.0,
    )
    scene.add(ris)

    for u in range(NUM_USERS):
        rx = Receiver(
            name=f"USER_{u + 1}",
            position=INITIAL_USER_POSITIONS[u],
            look_at=RIS_POSITION,
        )
        scene.add(rx)

    return scene, tx, ris


def verify_dimensions(scene, ris):
    print("\nDIMENSION VERIFICATION")
    print("-" * 82)

    bs_ant = int(scene.tx_array.num_ant)
    ris_cells = int(ris.num_cells)

    print(f"Configured BS antennas : {bs_ant}")
    print(f"Configured RIS cells   : {ris_cells}")

    if bs_ant != BS_ANTENNAS:
        raise RuntimeError(
            f"BS antenna mismatch: expected {BS_ANTENNAS}, got {bs_ant}"
        )

    if ris_cells != RIS_ELEMENTS:
        raise RuntimeError(
            f"RIS element mismatch: expected {RIS_ELEMENTS}, got {ris_cells}"
        )

    print("PASS: BS = 4 antennas")
    print("PASS: RIS = 64 elements")


# ============================================================
# DIRECT BS -> USER
# ============================================================

def trace_bs_user(scene, user_positions):
    print("\nTracing BS -> USER direct channels...")

    all_users = []

    for t in range(TIME_STEPS):
        for u in range(NUM_USERS):
            scene.get(f"USER_{u + 1}").position = user_positions[t, u]

        paths = scene.compute_paths(
            los=True,
            reflection=True,
            diffraction=False,
            scattering=False,
            ris=False,
            check_scene=True,
        )

        a, _ = paths.cir(
            los=True,
            reflection=True,
            diffraction=False,
            scattering=False,
            ris=False,
            cluster_ris_paths=False,
        )

        x = a.numpy()

        # Sionna:
        # [batch, rx, rx_ant, tx, tx_ant, paths, time]
        require_shape_rank(x, 7, "BS-user CIR")

        # For this setup:
        # [1, 2, 1, 1, 4, P, 1]
        x = squeeze_singleton_axes(
            x,
            (0, 2, 3, 6),
            "BS-user CIR",
        )

        # [2, 4, P]
        h = coherent_sum_paths(x, path_axis=-1)

        if h.shape != (NUM_USERS, BS_ANTENNAS):
            raise RuntimeError(
                f"Wrong BS-user intermediate shape: {h.shape}"
            )

        all_users.append(
            h.astype(np.complex64)
        )

    result = np.stack(all_users, axis=1)

    expected = (
        NUM_USERS,
        TIME_STEPS,
        BS_ANTENNAS,
    )

    print(f"true_csi_bs_user : {result.shape}")

    if result.shape != expected:
        raise RuntimeError(
            f"BS-user CSI shape error. Expected {expected}, got {result.shape}"
        )

    return result


# ============================================================
# BS -> RIS ELEMENT CSI
# ============================================================

def create_bs_ris_segment_scene():
    scene = load_scene(sionna.rt.scene.simple_street_canyon)

    scene.frequency = FREQUENCY_HZ
    scene.bandwidth = BANDWIDTH_HZ

    # BS: 4 TX antennas
    scene.tx_array = PlanarArray(
        num_rows=BS_ROWS,
        num_cols=BS_COLS,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    # RIS represented by 64 physical array ports.
    # This is used to extract element-wise ray-traced CSI.
    scene.rx_array = PlanarArray(
        num_rows=RIS_ROWS,
        num_cols=RIS_COLS,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    tx = Transmitter(
        name="BS_SEGMENT",
        position=BS_POSITION,
        look_at=RIS_POSITION,
        power_dbm=TX_POWER_DBM,
    )

    rx = Receiver(
        name="RIS_ELEMENTS",
        position=RIS_POSITION,
        look_at=BS_POSITION,
    )

    scene.add(tx)
    scene.add(rx)

    return scene


def trace_bs_ris():
    print("\nTracing BS -> RIS element CSI...")

    scene = create_bs_ris_segment_scene()

    paths = scene.compute_paths(
        los=True,
        reflection=True,
        diffraction=False,
        scattering=False,
        ris=False,
        check_scene=True,
    )

    a, _ = paths.cir(
        los=True,
        reflection=True,
        diffraction=False,
        scattering=False,
        ris=False,
        cluster_ris_paths=False,
    )

    x = a.numpy()

    # Expected:
    # [1, 1, 64, 1, 4, P, 1]
    #
    # IMPORTANT:
    # axis 2 is the 64-element RIS dimension.
    # Therefore axis 2 MUST NOT be squeezed.
    require_shape_rank(x, 7, "BS-RIS CIR")

    x = squeeze_singleton_axes(
        x,
        (0, 1, 3, 6),
        "BS-RIS CIR",
    )

    # [64, 4, P]
    h = coherent_sum_paths(x, path_axis=-1)

    expected_one = (
        RIS_ELEMENTS,
        BS_ANTENNAS,
    )

    if h.shape != expected_one:
        raise RuntimeError(
            f"BS-RIS CSI shape error. Expected {expected_one}, got {h.shape}"
        )

    # BS and RIS are fixed, so repeat over the 5 temporal positions.
    h_time = np.repeat(
        h[np.newaxis, :, :],
        TIME_STEPS,
        axis=0,
    ).astype(np.complex64)

    expected = (
        TIME_STEPS,
        RIS_ELEMENTS,
        BS_ANTENNAS,
    )

    print(f"true_csi_bs_ris  : {h_time.shape}")

    if h_time.shape != expected:
        raise RuntimeError(
            f"BS-RIS CSI shape error. Expected {expected}, got {h_time.shape}"
        )

    return h_time


# ============================================================
# RIS -> USER ELEMENT CSI
# ============================================================

def create_ris_user_segment_scene(user_position):
    scene = load_scene(sionna.rt.scene.simple_street_canyon)

    scene.frequency = FREQUENCY_HZ
    scene.bandwidth = BANDWIDTH_HZ

    # 64 RIS element ports as TX array
    scene.tx_array = PlanarArray(
        num_rows=RIS_ROWS,
        num_cols=RIS_COLS,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    tx = Transmitter(
        name="RIS_ELEMENTS",
        position=RIS_POSITION,
        look_at=user_position,
        power_dbm=TX_POWER_DBM,
    )

    rx = Receiver(
        name="USER",
        position=user_position,
        look_at=RIS_POSITION,
    )

    scene.add(tx)
    scene.add(rx)

    return scene


def trace_ris_user(user_positions):
    print("\nTracing RIS -> USER element CSI...")

    result = []

    for u in range(NUM_USERS):
        user_time = []

        for t in range(TIME_STEPS):
            scene = create_ris_user_segment_scene(
                user_positions[t, u]
            )

            paths = scene.compute_paths(
                los=True,
                reflection=True,
                diffraction=False,
                scattering=False,
                ris=False,
                check_scene=True,
            )

            a, _ = paths.cir(
                los=True,
                reflection=True,
                diffraction=False,
                scattering=False,
                ris=False,
                cluster_ris_paths=False,
            )

            x = a.numpy()

            # Expected:
            # [1, 1, 1, 1, 64, P, 1]
            require_shape_rank(x, 7, "RIS-user CIR")

            x = squeeze_singleton_axes(
                x,
                (0, 1, 2, 3, 6),
                "RIS-user CIR",
            )

            # [64, P]
            h = coherent_sum_paths(x, path_axis=-1)

            if h.shape != (RIS_ELEMENTS,):
                raise RuntimeError(
                    f"RIS-user CSI intermediate shape error: {h.shape}"
                )

            user_time.append(
                h.astype(np.complex64)
            )

        result.append(
            np.stack(user_time, axis=0)
        )

    result = np.stack(result, axis=0)

    expected = (
        NUM_USERS,
        TIME_STEPS,
        RIS_ELEMENTS,
    )

    print(f"true_csi_ris_user : {result.shape}")

    if result.shape != expected:
        raise RuntimeError(
            f"RIS-user CSI shape error. Expected {expected}, got {result.shape}"
        )

    return result


# ============================================================
# ACTUAL SIONNA RIS=True VERIFICATION
# ============================================================

def trace_actual_ris_support(scene):
    print("\nActual Sionna RIS support check (ris=True)...")

    paths = scene.compute_paths(
        los=False,
        reflection=False,
        diffraction=False,
        scattering=False,
        ris=True,
        check_scene=True,
    )

    a, tau = paths.cir(
        los=False,
        reflection=False,
        diffraction=False,
        scattering=False,
        ris=True,
        cluster_ris_paths=False,
    )

    print(f"Actual RIS CIR shape   : {tuple(a.shape)}")
    print(f"Actual RIS delay shape : {tuple(tau.shape)}")

    if a.shape.rank != 7:
        raise RuntimeError(
            f"Unexpected actual RIS CIR rank: {a.shape}"
        )

    return a, tau


# ============================================================
# CASCADED RIS CHANNEL
# ============================================================

def effective_ris_channel(
    h_bs_ris: tf.Tensor,
    h_ris_user: tf.Tensor,
    phase: tf.Tensor,
    h_bs_user: tf.Tensor | None = None,
) -> tf.Tensor:
    """Compute direct + RIS-assisted BS->user channel.

    Shapes (physical meaning):
      H_bs_ris   = [time, RIS_element, BS_antenna] = [5,64,4]
      H_ris_user = [user, time, RIS_element]       = [2,5,64]
      H_bs_user  = [user, time, BS_antenna]         = [2,5,4]
      phase      = [RIS_element]                    = [64]

    The RIS cascade is formed element-by-element and summed only over the
    64 RIS elements. No reshape/squeeze is used to hide dimensions.
    """
    h_br = tf.convert_to_tensor(h_bs_ris, dtype=tf.complex64)
    h_ru = tf.convert_to_tensor(h_ris_user, dtype=tf.complex64)
    phi = tf.convert_to_tensor(phase, dtype=tf.float32)

    tf.debugging.assert_equal(tf.shape(h_br),
                              [TIME_STEPS, RIS_ELEMENTS, BS_ANTENNAS],
                              message="H_bs_ris must be [5,64,4]")
    tf.debugging.assert_equal(tf.shape(h_ru),
                              [NUM_USERS, TIME_STEPS, RIS_ELEMENTS],
                              message="H_ris_user must be [2,5,64]")
    tf.debugging.assert_equal(tf.size(phi), RIS_ELEMENTS,
                              message="RIS phase vector must contain 64 values")

    # e^(j phi), kept complex so the phase cannot be discarded.
    reflection = tf.exp(tf.complex(tf.zeros_like(phi), phi))

    # [2,5,64,1] * [1,1,64,1] * [1,5,64,4]
    h_ru_4d = h_ru[..., tf.newaxis]
    phase_4d = reflection[tf.newaxis, tf.newaxis, :, tf.newaxis]
    h_br_4d = h_br[tf.newaxis, ...]

    cascaded = h_ru_4d * phase_4d * h_br_4d
    cascaded = tf.reduce_sum(cascaded, axis=2)  # [2,5,4]

    tf.debugging.assert_equal(tf.shape(cascaded),
                              [NUM_USERS, TIME_STEPS, BS_ANTENNAS],
                              message="RIS cascade must be [2,5,4]")

    if h_bs_user is None:
        return cascaded

    h_direct = tf.convert_to_tensor(h_bs_user, dtype=tf.complex64)
    tf.debugging.assert_equal(tf.shape(h_direct),
                              [NUM_USERS, TIME_STEPS, BS_ANTENNAS],
                              message="H_bs_user must be [2,5,4]")

    return h_direct + cascaded


# ============================================================
# ZF SINR / SUM RATE
# ============================================================

def zf_metrics(
    h_user_bs: tf.Tensor,
    noise_power_w: tf.Tensor | float,
):
    """
    h_user_bs:
        [U, M] complex channel

    Equal total TX power is split over U users.

    ZF beamforming is used because BS has 4 antennas and
    there are 2 users.

    Returns:
        rx_power [U]
        sinr [U]
        sum_rate scalar
    """
    h = tf.cast(h_user_bs, tf.complex64)

    num_users = tf.shape(h)[0]
    num_ant = tf.shape(h)[1]

    # H: [U,M]
    hh = tf.matmul(
        h,
        h,
        adjoint_b=True,
    )

    reg = tf.cast(
        1e-9,
        tf.complex64,
    ) * tf.eye(
        num_users,
        dtype=tf.complex64,
    )

    inv = tf.linalg.inv(hh + reg)

    # W: [M,U]
    w = tf.matmul(
        h,
        inv,
        adjoint_a=True,
    )

    # Normalize each beam
    w_norm = tf.sqrt(
        tf.reduce_sum(
            tf.abs(w) ** 2,
            axis=0,
            keepdims=True,
        )
        + tf.cast(EPS, tf.float32)
    )

    w = w / tf.cast(w_norm, tf.complex64)

    tx_power = tf.cast(
        dbm_to_watt(TX_POWER_DBM),
        tf.float32,
    )

    power_per_user = tx_power / tf.cast(
        num_users,
        tf.float32,
    )

    # Effective multi-user channel:
    # [U,M] @ [M,U] -> [U,U]
    effective = tf.matmul(
        h,
        w,
    )

    signal = tf.abs(
        tf.linalg.diag_part(effective)
    ) ** 2

    total = tf.abs(effective) ** 2

    interference = (
        tf.reduce_sum(
            total,
            axis=1,
        )
        - signal
    )

    signal_power = (
        power_per_user
        * signal
    )

    interference_power = (
        power_per_user
        * tf.maximum(
            interference,
            0.0,
        )
    )

    noise = tf.cast(
        noise_power_w,
        tf.float32,
    )

    sinr = signal_power / (
        interference_power
        + noise
        + EPS
    )

    rate = (
        tf.cast(BANDWIDTH_HZ, tf.float32)
        * tf.math.log1p(sinr)
        / tf.math.log(
            tf.constant(2.0, tf.float32)
        )
    )

    sum_rate = tf.reduce_sum(rate)

    return (
        signal_power,
        sinr,
        sum_rate,
    )


def evaluate_cascaded_phase(
    h_bs_ris,
    h_ris_user,
    h_bs_user,
    phase_np,
    noise_power_w,
):
    """Evaluate direct + RIS-assisted channel for one RIS phase profile."""
    phase = tf.convert_to_tensor(
        phase_np,
        dtype=tf.float32,
    )

    h_eff = effective_ris_channel(
        tf.convert_to_tensor(h_bs_ris),
        tf.convert_to_tensor(h_ris_user),
        phase,
        tf.convert_to_tensor(h_bs_user),
    )

    # First temporal state for the deterministic phase sanity test.
    h0 = h_eff[:, 0, :]

    rx_power, sinr, sum_rate = zf_metrics(
        h0,
        tf.cast(noise_power_w, tf.float32),
    )

    return {
        "h_eff": h_eff.numpy(),
        "rx_power_w": rx_power.numpy().astype(np.float32),
        "rx_power_dbm": np.array(
            [watt_to_dbm(x) for x in rx_power.numpy()],
            dtype=np.float32,
        ),
        "sinr": sinr.numpy().astype(np.float32),
        "sinr_db": (
            10.0
            * np.log10(
                np.maximum(
                    sinr.numpy(),
                    EPS,
                )
            )
        ).astype(np.float32),
        "sum_rate_bps": float(sum_rate.numpy()),
    }


# ============================================================
# NOISE CALIBRATION
# ============================================================

def calibrate_noise_power(
    h_bs_ris,
    h_ris_user,
    h_bs_user,
    reference_phase,
):
    """Calibrate AWGN so the reference state has the configured SNR."""
    phase = tf.convert_to_tensor(
        reference_phase,
        dtype=tf.float32,
    )

    h_eff = effective_ris_channel(
        tf.convert_to_tensor(h_bs_ris),
        tf.convert_to_tensor(h_ris_user),
        phase,
        tf.convert_to_tensor(h_bs_user),
    )

    h0 = h_eff[:, 0, :]

    ref_rx_power, _, _ = zf_metrics(
        h0,
        tf.constant(1e-30, tf.float32),
    )

    reference_power = float(
        tf.reduce_mean(ref_rx_power).numpy()
    )

    target_linear_snr = 10.0 ** (SNR_DB / 10.0)
    noise_power = reference_power / target_linear_snr

    return max(float(noise_power), 1e-30)


# ============================================================
# RIS OPTIMIZATION ON RAY-TRACED CSI
# ============================================================

def _sum_rate_numpy(
    h_bs_ris_np,
    h_ris_user_np,
    h_bs_user_np,
    phase_np,
    noise_power_w,
):
    result = evaluate_cascaded_phase(
        h_bs_ris_np,
        h_ris_user_np,
        h_bs_user_np,
        phase_np,
        noise_power_w,
    )
    return float(result["sum_rate_bps"])


def coordinate_refine_ris(
    h_bs_ris_np,
    h_ris_user_np,
    h_bs_user_np,
    initial_phase,
    noise_power_w,
):
    """Deterministic coordinate refinement used as a numerical safeguard."""
    phase = wrap_phase_np(np.asarray(initial_phase, dtype=np.float32)).copy()
    best_rate = _sum_rate_numpy(
        h_bs_ris_np, h_ris_user_np, h_bs_user_np, phase, noise_power_w
    )

    candidate_offsets = np.array(
        [0.0, 0.5*np.pi, -0.5*np.pi, np.pi],
        dtype=np.float32,
    )

    for sweep in range(2):
        improved = False
        for k in range(RIS_ELEMENTS):
            original = float(phase[k])
            local_best = best_rate
            local_phase = original

            for delta in candidate_offsets:
                candidate = phase.copy()
                candidate[k] = wrap_phase_np(np.array([original + float(delta)]))[0]
                rate = _sum_rate_numpy(
                    h_bs_ris_np,
                    h_ris_user_np,
                    h_bs_user_np,
                    candidate,
                    noise_power_w,
                )
                if rate > local_best + 1e-6:
                    local_best = rate
                    local_phase = float(candidate[k])

            if local_best > best_rate + 1e-6:
                phase[k] = local_phase
                best_rate = local_best
                improved = True

        print(
            f"  coordinate sweep {sweep+1}: "
            f"{best_rate/1e6:.6f} Mbps"
        )
        if not improved:
            break

    return wrap_phase_np(phase).astype(np.float32), best_rate


def optimize_ris_from_raytraced_csi(
    h_bs_ris_np,
    h_ris_user_np,
    h_bs_user_np,
    noise_power_w,
):
    """Optimize the 64 RIS phases using the full direct+RIS channel."""
    print("\nRunning RIS phase optimization on full ray-traced CSI...")

    h_bs_ris = tf.convert_to_tensor(h_bs_ris_np, dtype=tf.complex64)
    h_ris_user = tf.convert_to_tensor(h_ris_user_np, dtype=tf.complex64)
    h_bs_user = tf.convert_to_tensor(h_bs_user_np, dtype=tf.complex64)

    best_rate = -np.inf
    best_phase = None
    best_history = None

    for restart in range(OPT_RESTARTS):
        print(f"  restart {restart + 1}/{OPT_RESTARTS}")

        if restart == 0:
            initial = np.zeros(RIS_ELEMENTS, dtype=np.float32)
        else:
            initial = np.random.uniform(
                -np.pi, np.pi, RIS_ELEMENTS
            ).astype(np.float32)

        phase = tf.Variable(initial, dtype=tf.float32, trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        history = []

        for iteration in range(OPT_ITERS):
            with tf.GradientTape() as tape:
                h_eff = effective_ris_channel(
                    h_bs_ris,
                    h_ris_user,
                    phase,
                    h_bs_user,
                )
                h0 = h_eff[:, 0, :]
                _, sinr, sum_rate = zf_metrics(
                    h0,
                    tf.cast(noise_power_w, tf.float32),
                )
                loss = -sum_rate / tf.cast(BANDWIDTH_HZ, tf.float32)

            gradient = tape.gradient(loss, phase)
            if gradient is None:
                raise RuntimeError("RIS optimizer gradient is None.")
            if not bool(tf.reduce_all(tf.math.is_finite(gradient)).numpy()):
                raise RuntimeError("RIS optimizer gradient contains NaN/Inf.")

            grad_norm = float(tf.linalg.global_norm([gradient]).numpy())
            optimizer.apply_gradients([(gradient, phase)])
            phase.assign(wrap_phase_tf(phase))

            current_rate = float(sum_rate.numpy())
            history.append([iteration, current_rate, grad_norm])

            if current_rate > best_rate:
                best_rate = current_rate
                best_phase = phase.numpy().copy()
                best_history = np.asarray(history, dtype=np.float64)

            if iteration == 0 or iteration % 10 == 0 or iteration == OPT_ITERS - 1:
                print(
                    f"    iter={iteration:03d} "
                    f"rate={current_rate/1e6:.6f} Mbps "
                    f"grad={grad_norm:.3e}"
                )

    if best_phase is None:
        raise RuntimeError("RIS optimizer produced no phase solution.")

    # Numerical safeguard: optimize the actual scalar objective directly.
    refined_phase, refined_rate = coordinate_refine_ris(
        h_bs_ris_np,
        h_ris_user_np,
        h_bs_user_np,
        best_phase,
        noise_power_w,
    )

    if refined_rate > best_rate + 1e-6:
        best_phase = refined_phase
        best_rate = refined_rate
        print(f"  refined continuous rate: {best_rate/1e6:.6f} Mbps")
    else:
        # Also test a zero-phase start so the optimizer cannot accidentally
        # return a numerically worse solution than the baseline.
        zero_phase = np.zeros(RIS_ELEMENTS, dtype=np.float32)
        zero_rate = _sum_rate_numpy(
            h_bs_ris_np, h_ris_user_np, h_bs_user_np, zero_phase, noise_power_w
        )
        if zero_rate > best_rate:
            best_phase, best_rate = coordinate_refine_ris(
                h_bs_ris_np,
                h_ris_user_np,
                h_bs_user_np,
                zero_phase,
                noise_power_w,
            )

    return wrap_phase_np(best_phase).astype(np.float32), best_history


# ============================================================
# PHASE QUANTIZATION
# ============================================================

def quantize_phase(
    phase,
    bits,
):
    levels = 2 ** int(bits)

    step = (
        2.0
        * np.pi
        / levels
    )

    wrapped = wrap_phase_np(
        np.asarray(phase)
    )

    quantized = (
        np.round(
            wrapped / step
        )
        * step
    )

    return wrap_phase_np(
        quantized
    ).astype(np.float32)


# ============================================================
# CSI ERROR
# ============================================================

def add_csi_error(
    csi,
    error_percent,
):
    sigma = float(error_percent) / 100.0

    noise = (
        np.random.normal(
            size=csi.shape
        )
        + 1j
        * np.random.normal(
            size=csi.shape
        )
    ).astype(np.complex64)

    noise /= np.sqrt(2.0)

    scale = max(
        float(
            np.mean(
                np.abs(csi)
            )
        ),
        1e-12,
    )

    return (
        csi
        + scale
        * sigma
        * noise
    ).astype(np.complex64)


# ============================================================
# 3D ENGINEERING VISUALIZATION
# ============================================================

def draw_box(ax, x0, y0, z0, dx, dy, dz):
    corners = [
        (x0, y0, z0),
        (x0 + dx, y0, z0),
        (x0 + dx, y0 + dy, z0),
        (x0, y0 + dy, z0),
    ]

    top = [
        (x, y, z0 + dz)
        for x, y, _ in corners
    ]

    for i in range(4):
        a = corners[i]
        b = corners[(i + 1) % 4]
        c = top[i]
        d = top[(i + 1) % 4]

        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            linewidth=1.0,
        )

        ax.plot(
            [c[0], d[0]],
            [c[1], d[1]],
            [c[2], d[2]],
            linewidth=1.0,
        )

        ax.plot(
            [a[0], c[0]],
            [a[1], c[1]],
            [a[2], c[2]],
            linewidth=0.8,
        )


def create_3d_visualization(user_positions):
    """Create a clean engineering-style 3D topology diagram.

    This function is visualization-only. It does NOT modify any ray tracing,
    CSI, RIS optimization, or HDF5 logic.
    """
    output = OUTPUT_DIR / "single_sample_3d_scene.png"

    fig = plt.figure(figsize=(15, 10), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # --------------------------------------------------------
    # Buildings: wireframe only so they never hide the topology
    # --------------------------------------------------------
    buildings = [
        (-12, -22, 0, 8, 10, 12),
        (5,   -22, 0, 10, 10, 18),
        (18,  -20, 0, 7,  9, 10),
        (-12,  13, 0, 8,  8, 14),
        (5,   15,  0, 10, 7, 12),
        (18,  13,  0, 7,  8, 16),
    ]

    def cuboid_wireframe(x, y, z, dx, dy, dz):
        pts = np.array([
            [x,      y,      z],
            [x+dx,   y,      z],
            [x+dx,   y+dy,   z],
            [x,      y+dy,   z],
            [x,      y,      z+dz],
            [x+dx,   y,      z+dz],
            [x+dx,   y+dy,   z+dz],
            [x,      y+dy,   z+dz],
        ])
        edges = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]
        for a, b in edges:
            ax.plot(
                [pts[a,0], pts[b,0]],
                [pts[a,1], pts[b,1]],
                [pts[a,2], pts[b,2]],
                linewidth=1.0,
                alpha=0.55,
            )

    for b in buildings:
        cuboid_wireframe(*b)

    # --------------------------------------------------------
    # Ground/grid reference
    # --------------------------------------------------------
    for x in np.arange(-30, 31, 5):
        ax.plot([x, x], [-25, 25], [0, 0], linewidth=0.35, alpha=0.25)
    for y in np.arange(-25, 26, 5):
        ax.plot([-30, 30], [y, y], [0, 0], linewidth=0.35, alpha=0.25)

    # --------------------------------------------------------
    # BS: 2x2 antenna array
    # --------------------------------------------------------
    ax.scatter(
        *BS_POSITION,
        s=260,
        marker="^",
        label="BS — 4 antennas",
        depthshade=False,
    )
    ax.text(
        BS_POSITION[0], BS_POSITION[1], BS_POSITION[2] + 1.8,
        "BS\n2×2 = 4 antennas",
        fontsize=10,
        ha="center",
        va="bottom",
    )

    # Small 2x2 antenna markers around BS
    bs_offsets = np.array([[-0.15,-0.15],[0.15,-0.15],[-0.15,0.15],[0.15,0.15]])
    for ox, oy in bs_offsets:
        ax.scatter(
            BS_POSITION[0] + ox,
            BS_POSITION[1] + oy,
            BS_POSITION[2],
            s=22,
            depthshade=False,
        )

    # --------------------------------------------------------
    # RIS: physically show all 64 elements as an 8x8 panel
    # --------------------------------------------------------
    wavelength = 299792458.0 / FREQUENCY_HZ
    spacing = 0.5 * wavelength
    offsets = (np.arange(RIS_ROWS) - (RIS_ROWS - 1)/2.0) * spacing

    for dy in offsets:
        for dz in offsets:
            ax.scatter(
                RIS_POSITION[0],
                RIS_POSITION[1] + dy,
                RIS_POSITION[2] + dz,
                s=18,
                marker="s",
                depthshade=False,
            )

    # RIS frame
    half_y = max(abs(offsets).max(), 0.15) + spacing/2
    half_z = max(abs(offsets).max(), 0.15) + spacing/2
    ax.plot(
        [RIS_POSITION[0], RIS_POSITION[0]],
        [RIS_POSITION[1]-half_y, RIS_POSITION[1]+half_y],
        [RIS_POSITION[2]-half_z, RIS_POSITION[2]-half_z],
        linewidth=2.0,
    )
    ax.plot(
        [RIS_POSITION[0], RIS_POSITION[0]],
        [RIS_POSITION[1]-half_y, RIS_POSITION[1]+half_y],
        [RIS_POSITION[2]+half_z, RIS_POSITION[2]+half_z],
        linewidth=2.0,
    )
    ax.plot(
        [RIS_POSITION[0], RIS_POSITION[0]],
        [RIS_POSITION[1]-half_y, RIS_POSITION[1]-half_y],
        [RIS_POSITION[2]-half_z, RIS_POSITION[2]+half_z],
        linewidth=2.0,
    )
    ax.plot(
        [RIS_POSITION[0], RIS_POSITION[0]],
        [RIS_POSITION[1]+half_y, RIS_POSITION[1]+half_y],
        [RIS_POSITION[2]-half_z, RIS_POSITION[2]+half_z],
        linewidth=2.0,
    )

    ax.text(
        RIS_POSITION[0], RIS_POSITION[1], RIS_POSITION[2] + half_z + 1.4,
        "RIS\n8×8 = 64 elements",
        fontsize=10,
        ha="center",
        va="bottom",
    )

    # --------------------------------------------------------
    # Users + mobility trajectories
    # --------------------------------------------------------
    for u in range(NUM_USERS):
        trajectory = np.asarray(user_positions[:, u, :], dtype=float)
        p = trajectory[0]

        ax.plot(
            trajectory[:,0], trajectory[:,1], trajectory[:,2],
            linestyle="-.", linewidth=2.2,
            label=f"User {u+1} trajectory",
        )
        ax.scatter(
            p[0], p[1], p[2],
            s=150, marker="o", depthshade=False,
        )
        ax.text(
            p[0], p[1], p[2] + 1.0,
            f"User {u+1}\nstart",
            fontsize=9,
            ha="center",
        )

        # End position
        q = trajectory[-1]
        ax.scatter(q[0], q[1], q[2], s=70, marker="x", depthshade=False)

        # BS -> User direct path
        ax.plot(
            [BS_POSITION[0], p[0]],
            [BS_POSITION[1], p[1]],
            [BS_POSITION[2], p[2]],
            linestyle="--", linewidth=1.0, alpha=0.65,
        )

        # RIS -> User path
        ax.plot(
            [RIS_POSITION[0], p[0]],
            [RIS_POSITION[1], p[1]],
            [RIS_POSITION[2], p[2]],
            linestyle=":", linewidth=1.4, alpha=0.75,
        )

    # BS -> RIS path
    ax.plot(
        [BS_POSITION[0], RIS_POSITION[0]],
        [BS_POSITION[1], RIS_POSITION[1]],
        [BS_POSITION[2], RIS_POSITION[2]],
        linewidth=2.0,
        linestyle="--",
        alpha=0.8,
        label="BS → RIS",
    )

    # --------------------------------------------------------
    # Axes / camera: intentionally elevated and wide so no wall
    # fills the camera view.
    # --------------------------------------------------------
    ax.set_title(
        "AURORA-RIS 6G — One-Sample 3D Wireless Environment",
        fontsize=16,
        pad=18,
    )
    ax.set_xlabel("X [m]", labelpad=8)
    ax.set_ylabel("Y [m]", labelpad=8)
    ax.set_zlabel("Z [m]", labelpad=8)

    ax.set_xlim(-30, 30)
    ax.set_ylim(-25, 25)
    ax.set_zlim(0, 22)
    ax.set_box_aspect((60, 50, 22))
    ax.view_init(elev=24, azim=-125)

    # Transparent panes + light grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.25)

    ax.legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
    )

    plt.tight_layout()
    plt.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"3D visualization saved: {output}")
    return output


# ============================================================
# HDF5 WRITER
# ============================================================

def write_h5(path: Path, data: dict):
    if path.exists():
        path.unlink()

    with h5py.File(path, "w") as f:
        f.attrs["project"] = "AURORA-RIS 6G"
        f.attrs["dataset_stage"] = "one-sample-verification"
        f.attrs["sionna_version"] = str(sionna.__version__)
        f.attrs["frequency_hz"] = FREQUENCY_HZ
        f.attrs["bandwidth_hz"] = BANDWIDTH_HZ
        f.attrs["bs_rows"] = BS_ROWS
        f.attrs["bs_cols"] = BS_COLS
        f.attrs["bs_antennas"] = BS_ANTENNAS
        f.attrs["ris_rows"] = RIS_ROWS
        f.attrs["ris_cols"] = RIS_COLS
        f.attrs["ris_elements"] = RIS_ELEMENTS
        f.attrs["num_users"] = NUM_USERS
        f.attrs["time_steps"] = TIME_STEPS

        for name, value in data.items():
            array = np.asarray(value)

            # HDF5 scalar datasets cannot use compression/chunk filters.
            if array.ndim == 0:
                f.create_dataset(
                    name,
                    data=array,
                )
            else:
                f.create_dataset(
                    name,
                    data=array,
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )


# ============================================================
# HDF5 INSPECTION
# ============================================================

def inspect_h5(path: Path):
    print("\n" + "=" * 82)
    print(f"H5 CONTENTS: {path}")
    print("=" * 82)

    with h5py.File(path, "r") as f:
        print("\nATTRIBUTES")
        for key, value in f.attrs.items():
            print(f"{key:28s}: {value}")

        print("\nDATASETS")
        for name in f.keys():
            ds = f[name]
            print(
                f"{name:34s} "
                f"shape={str(ds.shape):20s} "
                f"dtype={ds.dtype}"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print_configuration()

    # --------------------------------------------------------
    # 1. User mobility
    # --------------------------------------------------------
    user_positions, user_velocities = (
        make_user_trajectories()
    )

    # --------------------------------------------------------
    # 2. Create actual Sionna RIS scene
    # --------------------------------------------------------
    print("\n[1/8] Creating actual Sionna RIS scene...")

    scene, tx, ris = create_main_scene()

    verify_dimensions(
        scene,
        ris,
    )

    # Actual RIS=True trace verification
    actual_ris_cir, actual_ris_tau = (
        trace_actual_ris_support(scene)
    )

    # --------------------------------------------------------
    # 3. Direct BS -> User
    # --------------------------------------------------------
    print("\n[2/8] BS -> USER CSI...")
    true_csi_bs_user = trace_bs_user(
        scene,
        user_positions,
    )

    # --------------------------------------------------------
    # 4. BS -> RIS
    # --------------------------------------------------------
    print("\n[3/8] BS -> RIS CSI...")
    true_csi_bs_ris = trace_bs_ris()

    # --------------------------------------------------------
    # 5. RIS -> User
    # --------------------------------------------------------
    print("\n[4/8] RIS -> USER CSI...")
    true_csi_ris_user = trace_ris_user(
        user_positions,
    )

    # --------------------------------------------------------
    # Shape assertions
    # --------------------------------------------------------
    print("\nCHANNEL SHAPE VERIFICATION")
    print("-" * 82)

    print(
        "true_csi_bs_ris   :",
        true_csi_bs_ris.shape,
        " expected:",
        (TIME_STEPS, RIS_ELEMENTS, BS_ANTENNAS),
    )

    print(
        "true_csi_ris_user :",
        true_csi_ris_user.shape,
        " expected:",
        (NUM_USERS, TIME_STEPS, RIS_ELEMENTS),
    )

    print(
        "true_csi_bs_user  :",
        true_csi_bs_user.shape,
        " expected:",
        (NUM_USERS, TIME_STEPS, BS_ANTENNAS),
    )

    assert true_csi_bs_ris.shape == (
        TIME_STEPS,
        RIS_ELEMENTS,
        BS_ANTENNAS,
    )

    assert true_csi_ris_user.shape == (
        NUM_USERS,
        TIME_STEPS,
        RIS_ELEMENTS,
    )

    assert true_csi_bs_user.shape == (
        NUM_USERS,
        TIME_STEPS,
        BS_ANTENNAS,
    )

    # --------------------------------------------------------
    # 6. Noise calibration
    # --------------------------------------------------------
    print("\n[5/8] Calibrating SNR-controlled noise...")

    zero_phase = np.zeros(
        RIS_ELEMENTS,
        dtype=np.float32,
    )

    noise_power_w = calibrate_noise_power(
        true_csi_bs_ris,
        true_csi_ris_user,
        true_csi_bs_user,
        zero_phase,
    )

    print(
        f"Noise power : {noise_power_w:.6e} W"
    )
    print(
        f"Noise power : {watt_to_dbm(noise_power_w):.3f} dBm"
    )

    # --------------------------------------------------------
    # 7. RIS sanity + optimization
    # --------------------------------------------------------
    print("\n[6/8] RIS phase sanity + optimization...")

    random_phase = np.random.uniform(
        -np.pi,
        np.pi,
        RIS_ELEMENTS,
    ).astype(np.float32)

    # Directly verify the RIS term itself before adding the direct BS->USER path.
    zero_cascade = effective_ris_channel(
        tf.convert_to_tensor(true_csi_bs_ris),
        tf.convert_to_tensor(true_csi_ris_user),
        tf.convert_to_tensor(zero_phase),
        None,
    ).numpy()
    random_cascade = effective_ris_channel(
        tf.convert_to_tensor(true_csi_bs_ris),
        tf.convert_to_tensor(true_csi_ris_user),
        tf.convert_to_tensor(random_phase),
        None,
    ).numpy()
    cascade_diff = float(np.max(np.abs(zero_cascade - random_cascade)))
    print(f"RIS-only max |H_zero-H_random|: {cascade_diff:.6e}")
    if cascade_diff <= 1e-15:
        raise RuntimeError(
            "FAIL: RIS-only cascaded channel is phase-invariant. "
            "Check BS->RIS/RIS->USER CSI and phase multiplication."
        )

    zero_result = evaluate_cascaded_phase(
        true_csi_bs_ris,
        true_csi_ris_user,
        true_csi_bs_user,
        zero_phase,
        noise_power_w,
    )

    random_result = evaluate_cascaded_phase(
        true_csi_bs_ris,
        true_csi_ris_user,
        true_csi_bs_user,
        random_phase,
        noise_power_w,
    )

    optimized_phase, history = (
        optimize_ris_from_raytraced_csi(
            true_csi_bs_ris,
            true_csi_ris_user,
            true_csi_bs_user,
            noise_power_w,
        )
    )

    optimizer_result = evaluate_cascaded_phase(
        true_csi_bs_ris,
        true_csi_ris_user,
        true_csi_bs_user,
        optimized_phase,
        noise_power_w,
    )

    target_phase = quantize_phase(
        optimized_phase,
        PHASE_BITS,
    )

    quantized_result = evaluate_cascaded_phase(
        true_csi_bs_ris,
        true_csi_ris_user,
        true_csi_bs_user,
        target_phase,
        noise_power_w,
    )

    print("\n" + "=" * 82)
    print("RIS PHASE SANITY TEST — RAY-TRACED CASCADED CSI")
    print("=" * 82)

    for result in [
        zero_result,
        random_result,
        optimizer_result,
        quantized_result,
    ]:
        print(f"\n{result is zero_result and 'ZERO_PHASE' or ''}"
              f"{result is random_result and 'RANDOM_PHASE' or ''}"
              f"{result is optimizer_result and 'OPTIMIZER' or ''}"
              f"{result is quantized_result and 'QUANTIZED' or ''}")

        print(
            "Received power [dBm] :",
            np.round(
                result["rx_power_dbm"],
                6,
            ),
        )

        print(
            "SINR                :",
            np.round(
                result["sinr"],
                8,
            ),
        )

        print(
            "SINR [dB]           :",
            np.round(
                result["sinr_db"],
                4,
            ),
        )

        print(
            "Sum rate [Mbps]     :",
            result["sum_rate_bps"] / 1e6,
        )

    zero_rate = zero_result["sum_rate_bps"]
    random_rate = random_result["sum_rate_bps"]
    optimized_rate = optimizer_result["sum_rate_bps"]

    print("\n" + "=" * 82)
    print("OPTIMIZATION VALIDATION")
    print("=" * 82)

    print(
        f"Zero phase rate   : {zero_rate / 1e6:.6f} Mbps"
    )
    print(
        f"Random phase rate : {random_rate / 1e6:.6f} Mbps"
    )
    print(
        f"Optimized rate    : {optimized_rate / 1e6:.6f} Mbps"
    )
    print(
        f"Random - zero     : {(random_rate-zero_rate)/1e6:.6f} Mbps"
    )
    print(
        f"Optimized - zero  : {(optimized_rate-zero_rate)/1e6:.6f} Mbps"
    )

    h_shape = optimizer_result["h_eff"].shape
    print(f"Effective channel shape: {h_shape}  expected: {(NUM_USERS, TIME_STEPS, BS_ANTENNAS)}")
    if h_shape != (NUM_USERS, TIME_STEPS, BS_ANTENNAS):
        raise RuntimeError(f"FAIL: effective channel shape is {h_shape}")

    max_zero_random_diff = float(np.max(np.abs(
        zero_result["h_eff"] - random_result["h_eff"]
    )))
    max_zero_opt_diff = float(np.max(np.abs(
        zero_result["h_eff"] - optimizer_result["h_eff"]
    )))
    print(f"Max |H_zero - H_random|: {max_zero_random_diff:.6e}")
    print(f"Max |H_zero - H_opt|   : {max_zero_opt_diff:.6e}")

    if max_zero_random_diff <= 1e-15:
        raise RuntimeError("FAIL: RIS random phase does not change effective channel.")

    if optimized_rate <= zero_rate * (1.0 + 1e-6):
        raise RuntimeError(
            "FAIL: RIS optimizer did not improve over zero-phase baseline. "
            f"baseline={zero_rate:.6e}, optimized={optimized_rate:.6e}"
        )

    if np.allclose(
        random_result["rx_power_w"],
        zero_result["rx_power_w"],
        rtol=1e-5,
        atol=1e-15,
    ):
        raise RuntimeError(
            "FAIL: Random RIS phase did not change received power."
        )

    if np.allclose(
        random_phase,
        0.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            "FAIL: Random phase generation returned zero."
        )

    if np.allclose(
        optimized_phase,
        0.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "FAIL: Optimizer returned an all-zero phase vector."
        )

    if optimized_rate <= zero_rate:
        raise RuntimeError(
            "FAIL: Optimizer did not improve over zero-phase baseline."
        )

    print("PASS: Random phase changes received power.")
    print("PASS: Optimizer produces non-zero RIS phases.")
    print("PASS: Optimizer improves over zero-phase baseline.")

    # --------------------------------------------------------
    # Actual Sionna RIS phase perturbation sanity
    # --------------------------------------------------------
    print("\nActual Sionna RIS object phase perturbation check...")

    def set_actual_ris_phase(phase_np):
        values = tf.convert_to_tensor(
            phase_np.reshape(
                1,
                RIS_ROWS,
                RIS_COLS,
            ),
            dtype=tf.float32,
        )
        ris.phase_profile.values = values

    def actual_ris_received_power(phase_np):
        set_actual_ris_phase(phase_np)

        paths = scene.compute_paths(
            los=False,
            reflection=False,
            diffraction=False,
            scattering=False,
            ris=True,
            check_scene=True,
        )

        a, _ = paths.cir(
            los=False,
            reflection=False,
            diffraction=False,
            scattering=False,
            ris=True,
            cluster_ris_paths=False,
        )

        x = a.numpy()

        # [1,2,1,1,4,P,1] -> [2,4,P]
        x = squeeze_singleton_axes(
            x,
            (0, 2, 3, 6),
            "actual RIS CIR",
        )

        h = coherent_sum_paths(
            x,
            path_axis=-1,
        )

        # First user/time-independent phase sanity.
        power = np.sum(
            np.abs(h) ** 2,
            axis=1,
        )

        return power.astype(np.float32)

    actual_zero_power = actual_ris_received_power(
        zero_phase
    )

    actual_random_power = actual_ris_received_power(
        random_phase
    )

    print(
        "Actual Sionna RIS zero-phase power :",
        np.round(actual_zero_power, 12),
    )
    print(
        "Actual Sionna RIS random-phase power:",
        np.round(actual_random_power, 12),
    )

    if np.allclose(
        actual_zero_power,
        actual_random_power,
        rtol=1e-5,
        atol=1e-20,
    ):
        raise RuntimeError(
            "FAIL: Actual Sionna RIS phase perturbation did not "
            "change the ray-traced RIS channel."
        )

    print(
        "PASS: Sionna ris=True responds to RIS phase changes."
    )

    # Restore optimized phase on the actual RIS object.
    set_actual_ris_phase(
        optimized_phase
    )

    # --------------------------------------------------------
    # 8. Noisy CSI + H5 + visualization
    # --------------------------------------------------------
    print("\n[7/8] Creating estimated/noisy CSI...")

    csi_bs_ris = add_csi_error(
        true_csi_bs_ris,
        CSI_ERROR_PERCENT,
    )

    csi_ris_user = add_csi_error(
        true_csi_ris_user,
        CSI_ERROR_PERCENT,
    )

    csi_bs_user = add_csi_error(
        true_csi_bs_user,
        CSI_ERROR_PERCENT,
    )

    # One-sample verification:
    # no artificial blockage is injected here.
    # Sionna geometry/material interactions still determine ray visibility.
    blockage = np.zeros(
        (TIME_STEPS, NUM_USERS),
        dtype=np.int8,
    )

    # Dataset dictionary
    dataset = {
        # TRUE CSI
        "true_csi_bs_ris": true_csi_bs_ris,
        "true_csi_ris_user": true_csi_ris_user,
        "true_csi_bs_user": true_csi_bs_user,

        # ESTIMATED / NOISY CSI
        "csi_bs_ris": csi_bs_ris,
        "csi_ris_user": csi_ris_user,
        "csi_bs_user": csi_bs_user,

        # CONTEXT
        "user_position": user_positions,
        "user_velocity": user_velocities,
        "snr_db": np.float32(SNR_DB),
        "csi_error_pct": np.float32(CSI_ERROR_PERCENT),
        "blockage": blockage,
        "phase_resolution_bits": np.int8(PHASE_BITS),
        "noise_power_w": np.float32(noise_power_w),

        # RIS TARGET
        "continuous_optimal_phase": optimized_phase.astype(
            np.float32
        ),
        "target_phase": target_phase.astype(
            np.float32
        ),
        "target_phase_sin_cos": np.stack(
            [
                np.sin(target_phase),
                np.cos(target_phase),
            ],
            axis=-1,
        ).astype(np.float32),

        # PERFORMANCE
        "optimal_sinr": optimizer_result["sinr"],
        "optimal_sum_rate": np.float32(
            optimizer_result["sum_rate_bps"]
        ),
        "quantized_sum_rate": np.float32(
            quantized_result["sum_rate_bps"]
        ),
        "baseline_sum_rate": np.float32(
            zero_result["sum_rate_bps"]
        ),

        # SANITY
        "sanity_random_sum_rate": np.float32(
            random_result["sum_rate_bps"]
        ),
        "sanity_zero_rx_power_dbm": zero_result[
            "rx_power_dbm"
        ],
        "sanity_random_rx_power_dbm": random_result[
            "rx_power_dbm"
        ],
        "sanity_optimizer_rx_power_dbm": optimizer_result[
            "rx_power_dbm"
        ],
        "sanity_zero_sinr": zero_result[
            "sinr"
        ],
        "sanity_random_sinr": random_result[
            "sinr"
        ],
        "sanity_optimizer_sinr": optimizer_result[
            "sinr"
        ],

        # GEOMETRY
        "bs_position": BS_POSITION,
        "ris_position": RIS_POSITION,

        # EFFECTIVE CHANNELS
        "effective_channel_zero": zero_result[
            "h_eff"
        ][:, 0, :].astype(np.complex64),

        "effective_channel_random": random_result[
            "h_eff"
        ][:, 0, :].astype(np.complex64),

        "effective_channel_optimizer": optimizer_result[
            "h_eff"
        ][:, 0, :].astype(np.complex64),
    }

    print("\n[8/8] Writing ONE-SAMPLE H5 files...")

    train_path = OUTPUT_DIR / "train.h5"
    validation_path = OUTPUT_DIR / "validation.h5"
    test_path = OUTPUT_DIR / "test.h5"

    write_h5(
        train_path,
        dataset,
    )
    write_h5(
        validation_path,
        dataset,
    )
    write_h5(
        test_path,
        dataset,
    )

    # 3D scene visualization
    scene_path = create_3d_visualization(
        user_positions
    )

    # Optimization history
    history_path = (
        OUTPUT_DIR
        / "ris_optimization_history.csv"
    )

    np.savetxt(
        history_path,
        history,
        delimiter=",",
        header="iteration,sum_rate_bps,gradient_norm",
        comments="",
    )

    # Actual RIS shape metadata
    metadata = {
        "project": "AURORA-RIS 6G",
        "sionna_version": str(sionna.__version__),
        "frequency_hz": FREQUENCY_HZ,
        "bandwidth_hz": BANDWIDTH_HZ,
        "bs_antennas": BS_ANTENNAS,
        "ris_elements": RIS_ELEMENTS,
        "num_users": NUM_USERS,
        "time_steps": TIME_STEPS,
        "true_csi_bs_ris_shape": list(
            true_csi_bs_ris.shape
        ),
        "true_csi_ris_user_shape": list(
            true_csi_ris_user.shape
        ),
        "true_csi_bs_user_shape": list(
            true_csi_bs_user.shape
        ),
        "actual_sionna_ris_cir_shape": [
            int(x) for x in actual_ris_cir.shape
        ],
        "actual_sionna_ris_delay_shape": [
            int(x) for x in actual_ris_tau.shape
        ],
        "baseline_sum_rate_bps": zero_rate,
        "random_sum_rate_bps": random_rate,
        "optimized_sum_rate_bps": optimized_rate,
        "quantized_sum_rate_bps": float(
            quantized_result["sum_rate_bps"]
        ),
    }

    metadata_path = (
        OUTPUT_DIR
        / "single_sample_schema.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            metadata,
            fp,
            indent=2,
        )

    inspect_h5(train_path)

    print("\n" + "=" * 82)
    print("ONE-SAMPLE VERIFICATION COMPLETE")
    print("=" * 82)
    print(f"train.h5       : {train_path}")
    print(f"validation.h5  : {validation_path}")
    print(f"test.h5        : {test_path}")
    print(f"3D scene       : {scene_path}")
    print(f"optimization    : {history_path}")
    print(f"schema          : {metadata_path}")
    print("=" * 82)
    print("NO LARGE DATASET GENERATED.")
    print("VERIFY THIS SAMPLE BEFORE SCALING.")
    print("=" * 82)


if __name__ == "__main__":
    main()
