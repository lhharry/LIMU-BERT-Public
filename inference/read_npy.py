import argparse
from pathlib import Path

import numpy as np
from numpy.lib import format


DEFAULT_NPY_PATH = Path("dataset/motion/label_20_120.npy")


def format_preview(array: np.ndarray, max_rows: int, max_cols: int) -> str:
	if array.ndim == 0:
		return repr(array.item())

	if array.ndim == 1:
		preview = array[:max_rows]
	elif array.ndim == 2:
		preview = array[:max_rows, :max_cols]
	else:
		preview = array[:max_rows]

	return np.array2string(preview, threshold=max_rows * max_cols, edgeitems=3)


def print_array_summary(name: str, array: np.ndarray, max_rows: int, max_cols: int) -> None:
	print(f"[{name}]")
	print(f"  shape: {array.shape}")
	print(f"  dtype: {array.dtype}")
	print(f"  ndim: {array.ndim}")
	print(f"  size: {array.size}")

	if array.size == 0:
		print("  array is empty")
		return

	if np.issubdtype(array.dtype, np.number):
		numeric = array.astype(np.float64, copy=False)
		print(f"  min: {numeric.min():.6g}")
		print(f"  max: {numeric.max():.6g}")
		print(f"  mean: {numeric.mean():.6g}")
		print(f"  std: {numeric.std():.6g}")

	print("  preview:")
	print("  " + format_preview(array, max_rows=max_rows, max_cols=max_cols).replace("\n", "\n  "))


def load_npy_file(npy_path: Path, allow_pickle: bool) -> np.ndarray | np.lib.npyio.NpzFile:
	if not npy_path.exists():
		raise FileNotFoundError(f"File not found: {npy_path}")

	return np.load(npy_path, allow_pickle=allow_pickle)


def print_npy_header(npy_path: Path) -> None:
	with npy_path.open("rb") as file_handle:
		magic = format.read_magic(file_handle)
		if magic == (1, 0):
			shape, fortran_order, dtype = format.read_array_header_1_0(file_handle)
		elif magic == (2, 0):
			shape, fortran_order, dtype = format.read_array_header_2_0(file_handle)
		else:
			raise ValueError(f"Unsupported .npy version: {magic}")

		print("Header:")
		print(f"  version: {magic}")
		print(f"  shape: {shape}")
		print(f"  fortran_order: {fortran_order}")
		print(f"  dtype: {dtype}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Inspect the contents of a .npy or .npz file.")
	parser.add_argument("npy_path", nargs="?", default=str(DEFAULT_NPY_PATH), help="Path to the .npy file")
	parser.add_argument("--allow-pickle", action="store_true", help="Allow loading object arrays with pickle")
	parser.add_argument("--max-rows", type=int, default=5, help="Maximum preview rows to print")
	parser.add_argument("--max-cols", type=int, default=8, help="Maximum preview columns to print")
	args = parser.parse_args()

	npy_path = Path(args.npy_path)
	data = load_npy_file(npy_path, allow_pickle=args.allow_pickle)

	print(f"File: {npy_path}")

	if isinstance(data, np.lib.npyio.NpzFile):
		print("Type: npz archive")
		print(f"Keys: {list(data.files)}")
		#for key in data.files:
			#print_array_summary(key, data[key], args.max_rows, args.max_cols)
		#data.close()
		return

	print("Type: numpy array")
	print_npy_header(npy_path)
	print_array_summary("array", data, args.max_rows, args.max_cols)


if __name__ == "__main__":
	main()
