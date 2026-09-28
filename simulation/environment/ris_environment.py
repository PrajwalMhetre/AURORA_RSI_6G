import tensorflow as tf

from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    RIS,
)


FREQUENCY = 28e9


def main():

    print("=" * 60)
    print("AURORA-RIS | SIONNA 0.19.2")
    print("=" * 60)

    print(f"TensorFlow : {tf.__version__}")
    print(f"Frequency  : {FREQUENCY / 1e9:.1f} GHz")

    # ---------------------------------------------------------
    # 1. Create empty wireless scene
    # ---------------------------------------------------------
    scene = load_scene()

    scene.frequency = FREQUENCY

    # ---------------------------------------------------------
    # 2. Configure transmitter antenna
    # ---------------------------------------------------------
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    # ---------------------------------------------------------
    # 3. Configure receiver antenna
    # ---------------------------------------------------------
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    # ---------------------------------------------------------
    # 4. Base station / transmitter
    # ---------------------------------------------------------
    tx = Transmitter(
        name="base_station",
        position=[0.0, 0.0, 10.0],
        power_dbm=30.0,
    )

    # ---------------------------------------------------------
    # 5. User equipment / receiver
    # ---------------------------------------------------------
    rx = Receiver(
        name="user_1",
        position=[20.0, 10.0, 1.5],
    )

    # ---------------------------------------------------------
    # 6. RIS
    # ---------------------------------------------------------
    ris = RIS(
        name="ris_1",
        position=[10.0, 5.0, 5.0],
        num_rows=8,
        num_cols=8,
        num_modes=1,
    )

    # ---------------------------------------------------------
    # 7. Add devices to scene
    # ---------------------------------------------------------
    scene.add(tx)
    scene.add(rx)
    scene.add(ris)

    # Point TX towards RIS
    tx.look_at(ris.position)

    # Point RX towards RIS
    rx.look_at(ris.position)

    # ---------------------------------------------------------
    # 8. Configure RIS phase gradient
    # ---------------------------------------------------------
    ris.phase_gradient_reflector(
        rx.position,
        tx.position,
    )

    # ---------------------------------------------------------
    # 9. Compute propagation paths
    # ---------------------------------------------------------
    print("\nComputing propagation paths...")

    paths = scene.compute_paths(
        los=False,
        reflection=True,
        diffraction=False,
        scattering=False,
    )

    # ---------------------------------------------------------
    # 10. Basic results
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("SIMULATION SUCCESS")
    print("=" * 60)

    print(f"Frequency : {scene.frequency / 1e9:.2f} GHz")
    print(f"TX        : {tx.position}")
    print(f"RIS       : {ris.position}")
    print(f"RX        : {rx.position}")

    print("\nPropagation paths generated successfully.")

    # Channel impulse response
    cir = paths.cir()

    print(f"CIR shape : {cir[0].shape}")

    print("\nNext step:")
    print("1. Generate multiple RIS configurations")
    print("2. Extract CSI / channel response")
    print("3. Calculate received power / SINR")
    print("4. Save dataset")


if __name__ == "__main__":
    main()