#!/usr/bin/env python3
"""Convert raw PFSP instance files into a single Excel workbook.

This script reads all `.txt` files in the specified input directory,
parses them using the ``read_raw_instance`` function from
``pfsp.instance`` and writes an Excel workbook where each sheet
corresponds to one instance.  Sheet names are derived from the file
names (without the extension).

Usage
-----

```
python scripts/convert_instances.py --input-dir data/raw --output data/Instances.xlsx
```

Dependencies: pandas (with openpyxl engine) and numpy.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from pfsp.instance import read_raw_instance


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PFSP instances to Excel workbook")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing raw .txt instance files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output Excel file",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist or is not a directory")
    # Collect raw instance files
    txt_files = [p for p in input_dir.iterdir() if p.suffix.lower() == ".txt"]
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {input_dir}")
    # Create Excel writer
    # Use xlsxwriter engine to ensure a proper .xlsx zip file is produced.
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for txt_file in txt_files:
            instance = read_raw_instance(str(txt_file))
            m, n = instance.p_times.shape
            # Build DataFrame: header row and M rows of pairs
            header = pd.DataFrame([[m, n]])
            rows = []
            for i in range(m):
                row = []
                for j in range(n):
                    row.extend([j, instance.p_times[i, j]])
                rows.append(row)
            df = pd.concat([header, pd.DataFrame(rows)], ignore_index=True)
            sheet_name = instance.name
            # Excel sheet names cannot exceed 31 characters
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, header=False, index=False)
    print(f"Wrote {len(txt_files)} instances to {output_path}")


if __name__ == "__main__":
    main()