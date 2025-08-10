# consolidate_backtest_data.py
import os
import pandas as pd

DATA_DIR = "../backtest_data"
OUTPUT_FILE = "consolidated_5m_data.csv"


def consolidate_5m_files(data_dir, output_file):
    all_data = []

    for filename in os.listdir(data_dir):
        if filename.endswith("5m.csv"):
            symbol = filename.replace("_5m.csv", "")
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            df["ticker"] = symbol
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            all_data.append(df)

    combined_df = pd.concat(all_data).sort_values(by=["timestamp", "ticker"]).reset_index(drop=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Consolidated {len(all_data)} files into {output_file} with {len(combined_df)} rows.")


def main():
    consolidate_5m_files(DATA_DIR, OUTPUT_FILE)


if __name__ == "__main__":
    main()