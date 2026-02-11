import sys
from pathlib import Path

from qecdec import RotatedSurfaceCode_Memory

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.dataset.dataset_builder import build_decoding_datasets

    d_list = [5, 7, 9]
    p = 0.01

    data_dir = Path(__file__).resolve().parent.parent / "datasets" / "rotated_surface_code_memory_Z"

    for d in d_list:
        data_subdir = data_dir / f"d={d}_rounds={d}_p={p}"
        train_dataset_path = data_subdir / "train_dataset.pt"
        val_dataset_path = data_subdir / "val_dataset.pt"
        if train_dataset_path.exists() and val_dataset_path.exists():
            continue

        expmt = RotatedSurfaceCode_Memory(
            d=d,
            rounds=d,
            basis='Z',
            data_qubit_error_rate=p,
            meas_error_rate=p,
        )
        train_dataset, val_dataset = build_decoding_datasets(
            expmt,
            train_shots=10_000,
            val_shots=1_000,
            seed=42,
            incl_all_wt1_errors_in_train=True,
            incl_all_wt2_errors_in_train=True,
            remove_trivial_syndromes=True,
        )
        train_dataset.save_to_file(train_dataset_path)
        val_dataset.save_to_file(val_dataset_path)
