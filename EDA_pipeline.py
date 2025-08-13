import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns

def eda_pipeline(data_path, target_commodity):
    """
    EDA Pipeline for Kalimati dataset.
    Cleans data, removes duplicates, imputes missing dates (except start/end years),
    and generates basic exploratory plots.
    """

    # --- Load Data ---
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # --- Convert numeric columns ---
    for col in ['Minimum', 'Maximum', 'Average']:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace('Rs', '').str.strip(), errors='coerce')

    # --- Parse date ---
    def format_date(date):
        try:
            return parser.parse(date).strftime("%Y-%m-%d")
        except:
            return pd.NaT
    df['Date'] = pd.to_datetime(df['Date'].apply(format_date), errors='coerce')

    # --- Filter target commodity ---
    df = df[df['Commodity'] == target_commodity].copy()

    # --- Remove duplicates ---
    df = df.drop_duplicates(subset=['Date', 'Commodity'], keep='first').reset_index(drop=True)

    # --- Count before imputation ---
    yearly_counts_before = df.groupby(df['Date'].dt.year).size()

    # --- Impute missing dates (exclude 2013 & 2023) ---
    processed_data = []
    missing_dates_info = {}

    for year in range(2014, 2023):
        df_year = df[df['Date'].dt.year == year].copy()
        full_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
        missing_dates = full_dates.difference(df_year['Date'])
        missing_dates_info[year] = list(missing_dates)

        for missing_date in missing_dates:
            prev_day = missing_date - pd.Timedelta(days=1)
            prev_row = df_year[df_year['Date'] == prev_day]
            if not prev_row.empty:
                new_row = prev_row.copy()
                new_row['Date'] = missing_date
                df_year = pd.concat([df_year, new_row], ignore_index=True)

        df_year = df_year.sort_values('Date').reset_index(drop=True)
        processed_data.append(df_year)

    # Merge with unchanged 2013 & 2023
    final_df = pd.concat(
        [df[df['Date'].dt.year == 2013]] + processed_data + [df[df['Date'].dt.year == 2023]],
        ignore_index=True
    )

    # --- Yearly counts after ---
    yearly_counts_after = final_df.groupby(final_df['Date'].dt.year).size()

    # --- Missing dates after ---
    def find_missing_dates_per_year(df, start_year=2014, end_year=2022):
        missing_after = {}
        for year in range(start_year, end_year + 1):
            full_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
            present_dates = df[df['Date'].dt.year == year]['Date'].dt.normalize().unique()
            missing = set(full_range) - set(present_dates)
            missing_after[year] = sorted(missing)
        return missing_after

    missing_dates_after = find_missing_dates_per_year(final_df)

    # --- Print Summary ---
    print(f"\nEDA Summary for {target_commodity}")
    print(f"Initial shape: {df.shape}")
    print("\nYearly instance counts before imputation:")
    print(yearly_counts_before)
    print("\nMissing dates count BEFORE imputation:")
    for y, dates in missing_dates_info.items():
        print(f"{y}: {len(dates)}")
    print("\nMissing dates count AFTER imputation:")
    for y, dates in missing_dates_after.items():
        print(f"{y}: {len(dates)}")
    print("\nYearly instance counts after imputation:")
    print(yearly_counts_after)
    print(f"\nFinal shape after processing: {final_df.shape}")

    # # --- Visualizations ---
    # plt.figure(figsize=(14,5))
    # plt.plot(final_df['Date'], final_df['Average'], label='Average Price')
    # plt.title(f'{target_commodity} Price Over Time')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(8,4))
    # sns.histplot(final_df['Average'], bins=30, kde=True)
    # plt.title(f'{target_commodity} Average Price Distribution')
    # plt.show()

    # final_df['Year'] = final_df['Date'].dt.year
    # plt.figure(figsize=(12,6))
    # sns.boxplot(data=final_df, x='Year', y='Average')
    # plt.title(f'{target_commodity} Average Price Distribution by Year')
    # plt.xticks(rotation=45)
    # plt.show()

    return final_df

# --- Run Pipeline ---
DATA_PATH = r"C:\Users\acer\Desktop\Data_Science_Projects\Projects 2025\Forecasting_Prices\kalimati_dataset.csv"
TARGET_COMMODITY = "Onion Dry (Indian)"
processed_df = eda_pipeline(DATA_PATH, TARGET_COMMODITY)
