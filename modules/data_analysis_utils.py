import numpy as np
import pandas as pd

def describe_dataframe(df):
    """
    For each column in df:
      - If dtype is object or int64 or bool, list each unique value and its counts.
      - If dtype is float64, display min, Q1, median, Q3, and max.
      - Otherwise, handle accordingly (datetime, etc.).
    """
    for col in df.columns:
        col_type = df[col].dtype

        print(f"Column: {col}")
        print(f"  Data Type: {col_type}")

        if col == "anes_procedure_encounter_id_2273" or col == "unique_pt_id":
            print(f"  Number unique: {len(df[col].unique())}")

        elif col_type == 'object' or col_type == 'int64' or col_type == 'bool':
            # Show unique values and their counts
            value_counts = df[col].value_counts(dropna=False)
            print("  Value counts:")
            for val, count in value_counts.items():
                print(f"    {val}: {count}")

        elif col_type == 'float64':
            # Show min, Q1 (25%), median (50%), Q3 (75%), and max
            desc = df[col].describe(percentiles=[0.25, 0.5, 0.75])
            na_count = df[col].isna().sum()
            print("  Summary stats:")
            print(f"    NaN:    {na_count}")
            print(f"    Min:    {desc['min']}")
            print(f"    Q1:     {desc['25%']}")
            print(f"    Median: {desc['50%']}")
            print(f"    Q3:     {desc['75%']}")
            print(f"    Max:    {desc['max']}")

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Example handling for datetime columns
            print("  (Datetime column – no numeric summary or value counts shown.)")

        else:
            # Handle any other data types as needed
            print("  (No specific handling implemented for this data type.)")

        print("-" * 50)

def describe_as_tables(df):
    # Separate columns by dtype
    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        if col == "anes_procedure_encounter_id_2273" or col == "unique_pt_id":
            pass
        elif df[col].dtype == 'object' or df[col].dtype == 'int64' or df[col].dtype == 'bool':
            categorical_cols.append(col)
        elif df[col].dtype == 'float64':
            numeric_cols.append(col)
        else:
            # skip or handle datetime, etc. if desired
            pass

    # --- Build table for categorical variables ---
    cat_data = {}
    for col in categorical_cols:
        # Get value counts (including NaN as a separate category)
        vc = df[col].value_counts(dropna=False)
        # Convert value counts to a dict, or a formatted string
        vc_str = ", ".join(f"{val}: {count}" for val, count in vc.items())
        cat_data[col] = {
            'value_counts': vc_str
        }
    cat_df = pd.DataFrame(cat_data).T  # Transpose so rows = columns, col = 'value_counts'

    # --- Build table for numeric variables ---
    num_data = {}
    for col in numeric_cols:
        desc = df[col].describe(percentiles=[0.25, 0.5, 0.75])
        na_count = df[col].isna().sum()
        num_data[col] = {
            'count': desc['count'],
            'count_nan': na_count,
            'min': desc['min'],
            'Q1': desc['25%'],
            'median': desc['50%'],
            'Q3': desc['75%'],
            'max': desc['max']
        }
    num_df = pd.DataFrame(num_data).T  # Transpose so rows = columns

    return cat_df, num_df


def parse_value_counts_str(value_counts_str):
    """
    Convert a string like:
       "bwh: 44730, mgh: 26549, nwh: 22476, slm: 5680..."
    into a dict, e.g.:
       {"bwh": 44730, "mgh": 26549, "nwh": 22476, "slm": 5680}
    It will ignore trailing '...' and attempt to parse each value as float.
    """
    # Strip and remove trailing ellipsis (if present)
    value_counts_str = value_counts_str.strip()
    if value_counts_str.endswith("..."):
        value_counts_str = value_counts_str[:-3].strip()
    
    out_dict = {}
    # Split by commas
    items = [s.strip() for s in value_counts_str.split(",") if s.strip()]
    for item in items:
        # Split on the first colon only
        parts = item.split(":", 1)
        if len(parts) != 2:
            # If we can't split into exactly "key: value", skip
            continue
        key = parts[0].strip()
        val_str = parts[1].strip()
        # Attempt to parse numeric value
        try:
            val = float(val_str)
        except ValueError:
            # If not parseable, store NaN or skip
            val = float("nan")
        out_dict[key] = val
    return out_dict


def create_table_one(cat_table, num_table):
    """
    cat_table: 
        index = categorical variable names
        column "value_counts" = string describing categories & counts (to be parsed)
    num_table:
        index = numeric variable names
        columns include: ["count", "count_nan", "min", "Q1", "median", "Q3", "max", ...]
    """
    table_rows = []

    # 1) Numeric variables: median [Q1 - Q3]
    for var_name in num_table.index:
        median_val = num_table.loc[var_name, "median"]
        q1 = num_table.loc[var_name, "Q1"]
        q3 = num_table.loc[var_name, "Q3"]

        summary_str = f"{median_val:.2f} [{q1:.2f} - {q3:.2f}]"
        table_rows.append([var_name, summary_str])

    # 2) Categorical variables
    for var_name in cat_table.index:
        # 2a) Parse the "value_counts" string into a dict
        raw_str = cat_table.loc[var_name, "value_counts"]
        value_counts_dict = parse_value_counts_str(raw_str)

        # Compute total (excluding missing if you prefer)
        total_n = sum(value_counts_dict.values())

        # 2b) Check if binary (i.e., keys == {0,1} after parsing)
        keys_set = set(value_counts_dict.keys())
        
        # Convert keys from string->float->int if needed
        # (Because if your raw data had "1: 106750", then key might be "1" (string), or float(1.0).)
        # We can do a quick normalization:
        try:
            int_keys = {int(float(k)) for k in keys_set}
        except:
            int_keys = set()  # In case it fails

        if int_keys == {0, 1} and len(keys_set) == 2:
            # If it's truly binary: single row for the percent of '1'
            # (Need to fetch the count for '1' – might be string or float key)
            # We'll do a small loop to figure out which key is '1'
            n_ones = 0
            for k, v in value_counts_dict.items():
                try:
                    if int(float(k)) == 1:
                        n_ones = v
                        break
                except:
                    pass
            pct_ones = 100.0 * n_ones / total_n if total_n else 0.0
            summary_str = f"{int(n_ones)} ({pct_ones:.2f}%)"  # cast to int if you prefer
            table_rows.append([var_name, summary_str])  # cast to int if you prefer
        
        else:
            # Multi-category: separate row per category
            # Sort the keys in some consistent manner
            # We'll attempt to sort by the natural ordering of strings
            # (Alternatively, sort by numeric if your categories are numeric.)
            sorted_keys = sorted(value_counts_dict.keys(), key=str)
            
            for cat_val in sorted_keys:
                n_cat = value_counts_dict[cat_val]
                pct_cat = 100.0 * n_cat / total_n if total_n else 0.0
                summary_str = f"{int(n_cat)} ({pct_cat:.2f}%)"  # cast to int if you prefer
                row_label = f"{var_name} = {cat_val}"
                table_rows.append([row_label, summary_str])

    # Build final DataFrame
    table_one = pd.DataFrame(table_rows, columns=["Variable", "Summary"])
    return table_one
