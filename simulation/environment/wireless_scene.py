"""
AURORA-RSI-6G
Step 1: Wireless Environment Setup

This module:
1. Loads a Sionna RT urban wireless scene
2. Configures the carrier frequency
3. Configures BS and UE antenna arrays
4. Adds one base station
5. Adds two users
6. Computes propagation paths using ray tracing
"""

import sionna.rt

from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
)


# ============================================================
# SIMULATION CONFIGURATION
# ============================================================

CARRIER_FREQUENCY = 28e9       # 28 GHz
MAX_PATH_DEPTH = 5

BS_POSITION = [-33.0, 11.0, 32.0]

USER_1_POSITION = [27.0, -13.0, 1.5]

USER_2_POSITION = [10.0, -8.0, 1.5]


# ============================================================
# 1. LOAD WIRELESS ENVIRONMENT
# ============================================================

def create_wireless_scene():
    """
    Load the Sionna simple street-canyon environment.
    """

    scene = load_scene(
        sionna.rt.scene.simple_street_canyon,
    )

    return scene


# ============================================================
# 2. CONFIGURE WIRELESS SYSTEM
# ============================================================

def configure_wireless_system(scene):
    """
    Configure carrier frequency, antenna arrays,
    base station and wireless users.
    """

    # --------------------------------------------------------
    # Carrier frequency
    # --------------------------------------------------------

    scene.frequency = CARRIER_FREQUENCY

    # --------------------------------------------------------
    # Base-station antenna array
    # --------------------------------------------------------

    scene.tx_array = PlanarArray(
        num_rows=4,
        num_cols=4,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    # --------------------------------------------------------
    # User antenna array
    # --------------------------------------------------------

    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="V",
    )

    # --------------------------------------------------------
    # Base station
    # --------------------------------------------------------

    base_station = Transmitter(
        name="base_station",
        position=BS_POSITION,
    )

    scene.add(base_station)

    # --------------------------------------------------------
    # User 1
    # --------------------------------------------------------

    user_1 = Receiver(
        name="user_1",
        position=USER_1_POSITION,
    )

    scene.add(user_1)

    # --------------------------------------------------------
    # User 2
    # --------------------------------------------------------

    user_2 = Receiver(
        name="user_2",
        position=USER_2_POSITION,
    )

    scene.add(user_2)

    return scene


# ============================================================
# 3. RAY TRACING / PROPAGATION PATHS
# ============================================================

def compute_propagation_paths(scene):
    """
    Compute wireless propagation paths between
    the transmitter and receivers.
    """

    paths = scene.compute_paths(
        max_depth=MAX_PATH_DEPTH,
        los=True,
        reflection=True,
        diffraction=False,
        scattering=False,
    )

    return paths


# ============================================================
# 4. PRINT SCENE INFORMATION
# ============================================================

def print_scene_information(scene):
    """
    Print a readable summary of the wireless environment.
    """

    print("\n" + "=" * 70)
    print("WIRELESS ENVIRONMENT INFORMATION")
    print("=" * 70)

    print(f"Carrier frequency : {CARRIER_FREQUENCY / 1e9:.2f} GHz")

    print(f"Transmitters      : {len(scene.transmitters)}")

    print(f"Receivers         : {len(scene.receivers)}")

    print(f"Scene objects     : {len(scene.objects)}")

    print("\nTransmitters:")

    for name in scene.transmitters:
        print(f"  TX -> {name}")

    print("\nReceivers:")

    for name in scene.receivers:
        print(f"  RX -> {name}")

    print("\nScene objects:")

    for name in scene.objects:
        print(f"  OBJ -> {name}")


# ============================================================
# 5. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AURORA-RSI-6G")
    print("STEP 1 - WIRELESS ENVIRONMENT SETUP")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Load environment
    # --------------------------------------------------------

    print("\n[1/4] Loading Sionna wireless environment...")

    scene = create_wireless_scene()

    print("      SUCCESS: Environment loaded.")

    # --------------------------------------------------------
    # Step 2: Configure system
    # --------------------------------------------------------

    print("\n[2/4] Configuring wireless system...")

    configure_wireless_system(scene)

    print("      SUCCESS: Wireless system configured.")

    print(
        f"      Frequency : "
        f"{CARRIER_FREQUENCY / 1e9:.2f} GHz"
    )

    print(
        f"      BS        : "
        f"{BS_POSITION}"
    )

    print(
        f"      User 1    : "
        f"{USER_1_POSITION}"
    )

    print(
        f"      User 2    : "
        f"{USER_2_POSITION}"
    )

    # --------------------------------------------------------
    # Step 3: Scene information
    # --------------------------------------------------------

    print("\n[3/4] Inspecting environment...")

    print_scene_information(scene)

    # --------------------------------------------------------
    # Step 4: Ray tracing
    # --------------------------------------------------------

    print("\n[4/4] Running ray tracing...")

    paths = compute_propagation_paths(scene)

    print("      SUCCESS: Propagation paths calculated.")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 1 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nSimulation summary:")

    print(f"  Carrier frequency : {CARRIER_FREQUENCY / 1e9:.2f} GHz")
    print(f"  Base stations    : {len(scene.transmitters)}")
    print(f"  Users            : {len(scene.receivers)}")
    print(f"  Scene objects    : {len(scene.objects)}")
    print(f"  Ray depth        : {MAX_PATH_DEPTH}")

    print("\nPropagation paths object:")

    print(f"  Type             : {type(paths).__name__}")

    print("\nNEXT PROJECT STEP:")
    print("  STEP 2 -> RIS integration")
    print("  STEP 3 -> CSI extraction")
    print("  STEP 4 -> Classical RIS optimization")
    print("  STEP 5 -> Dataset generation")
    print("  STEP 6 -> GNN + GRU training")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()