import argparse

from pipeline import run_dataset_generation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-voxel", action="store_true")
    parser.add_argument("--enable-sdf", action="store_true")
    parser.add_argument("--noise", action="store_true", dest="add_noise_to_geom")
    parser.add_argument(
        "--geom-id-start",
        type=int,
        default=1,
        help="Start index for geom_id (default: 1).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("-----------------------------------")
    print("------- Generating ....... --------")
    print("-----------------------------------")

    try:
        args = parse_args()
        run_dataset_generation(
            args.enable_voxel,
            args.enable_sdf,
            args.add_noise_to_geom,
            args.geom_id_start,
        )
    except Exception as e:
        print(f" Error occurred: {e}")
        import traceback

        traceback.print_exc()
