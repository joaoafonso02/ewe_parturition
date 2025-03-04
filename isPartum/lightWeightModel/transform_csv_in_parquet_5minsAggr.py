import polars as pl
from pathlib import Path

# Define the directory containing the CSV files
csv_directory = Path(".")  
output_directory = Path("./parquets5mins") 
output_directory.mkdir(parents=True, exist_ok=True)

# Function to transform and compress CSV file using Parquet
def transform_and_compress_csv(file_path):
    try:
        # Read the CSV file using polars
        df = pl.read_csv(file_path, separator=';')

        print(f"Processing: {file_path}")
        print(df.head())

        if "Time" not in df.columns:
            raise ValueError("Missing 'Time' column in CSV.")

        df = df.with_columns(
            pl.col("Time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False)
        )

        print(df.dtypes)

        df = df.sort("Time")

        # Group by 5-minute intervals and compute multiple aggregations for sensor columns.
        aggregated_df = df.group_by_dynamic("Time", every="5m").agg([
            # Acc_X (mg) 
            pl.col("Acc_X (mg)").median().alias("Acc_X_median"),
            pl.col("Acc_X (mg)").mean().alias("Acc_X_mean"),
            pl.col("Acc_X (mg)").std().alias("Acc_X_std"),
            pl.col("Acc_X (mg)").max().alias("Acc_X_max"),
            pl.col("Acc_X (mg)").min().alias("Acc_X_min"),
            # Acc_Y (mg) 
            pl.col("Acc_Y (mg)").median().alias("Acc_Y_median"),
            pl.col("Acc_Y (mg)").mean().alias("Acc_Y_mean"),
            pl.col("Acc_Y (mg)").std().alias("Acc_Y_std"),
            pl.col("Acc_Y (mg)").max().alias("Acc_Y_max"),
            pl.col("Acc_Y (mg)").min().alias("Acc_Y_min"),
            # Acc_Z (mg) 
            pl.col("Acc_Z (mg)").median().alias("Acc_Z_median"),
            pl.col("Acc_Z (mg)").mean().alias("Acc_Z_mean"),
            pl.col("Acc_Z (mg)").std().alias("Acc_Z_std"),
            pl.col("Acc_Z (mg)").max().alias("Acc_Z_max"),
            pl.col("Acc_Z (mg)").min().alias("Acc_Z_min"),
            # Temperature (C) 
            pl.col("Temperature (C)").median().alias("Temperature_median"),
            pl.col("Temperature (C)").mean().alias("Temperature_mean"),
            pl.col("Temperature (C)").std().alias("Temperature_std"),
            pl.col("Temperature (C)").max().alias("Temperature_max"),
            pl.col("Temperature (C)").min().alias("Temperature_min"),
            # Class aggregation (keep median for label consistency)
            pl.col("Class").median().alias("Class")
        ])

        # Save to a Parquet file
        output_file = output_directory / (file_path.stem + ".parquet")
        aggregated_df.write_parquet(output_file)
        print(f"Transformed and compressed data saved to {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

for csv_file in csv_directory.glob("*.csv"):
    transform_and_compress_csv(csv_file)
