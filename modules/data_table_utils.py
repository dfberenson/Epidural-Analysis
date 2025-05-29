import numpy as np
import pandas as pd
from scipy import stats

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

        
        # summary_str = f"{median_val:.2f} [{q1:.2f}, {q3:.2f}] (NaN count: {num_table.loc[var_name, 'count_nan']})"
        summary_str = f"{median_val:.2f} [{q1:.2f}, {q3:.2f}]"
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
            summary_str = f"{int(n_ones)} ({pct_ones:.1f}%)"  # cast to int if you prefer
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
                summary_str = f"{int(n_cat)} ({pct_cat:.1f}%)"  # cast to int if you prefer
                row_label = f"{var_name} = {cat_val}"
                table_rows.append([row_label, summary_str])

    # Build final DataFrame
    table_one = pd.DataFrame(table_rows, columns=["Variable", "Summary"])
    return table_one


def clean_table_one(df):
    # Step 1: Clean variable names
    def clean_variable_name(name):
        manual_map = {
            "gestational_age_weeks": "Gestational Age (weeks)",
            "baby_weight_2196": "Baby Weight",
            "rom_thru_delivery_hours": "ROM to Delivery (hours)",
            "bmi_end_pregnancy_2044": "BMI at End of Pregnancy",
            "maternal_weight_end_pregnancy_2045": "Maternal Weight at End of Pregnancy",
            "bmi_before_pregnancy_2161": "BMI Before Pregnancy",
            "gravidity_2047": "Gravidity",
            "parity_2048": "Parity",
            "lor_depth": "Loss of Resistance Depth",
            "current_resident_catheter_count": "Resident Catheter Count (Current)",
            "total_team_catheter_count": "Team Catheter Count (Total)",
            "current_anesthesiologist_catheter_count": "Anesthesiologist Catheter Count (Current)",
            "prior_pain_scores_max": "Max Prior Pain Score",
            "prior_ob_cmi_scores_max": "Max OB CMI Score",
            "number_of_neuraxial_attempts": "Number of Neuraxial Attempts",
            "prior_failed_catheters_this_enc": "Prior Failed Catheters (This Encounter)",
            "prior_failed_catheters_prev_enc": "Prior Failed Catheters (Previous Encounter)",
            "prior_all_catheters_all_enc": "All Catheters (All Encounters)",
            "maternal_age_years": "Maternal Age",
            "placement_to_delivery_hours": "Placement to Delivery (hours)",
            "rom_to_placement_hours": "ROM to Placement (hours)",
            "delivery_site_bwh": "Delivery Site: BWH",
            "bmi_greater_than_40": "BMI > 40",
            "is_neuraxial_catheter = 1": "Neuraxial Catheter Present",
            "failed_catheter": "Failed Catheter",
            "has_subsequent_neuraxial_catheter": "Subsequent Neuraxial Catheter",
            "has_subsequent_spinal": "Subsequent Spinal Procedure",
            "has_subsequent_airway": "Subsequent Airway Procedure",
            "has_resident": "Resident Present",
            "has_anesthesiologist": "Anesthesiologist Present",
            "high_bmi_and_highly_experienced_resident": "High BMI + Experienced Resident",
            "high_bmi_and_lowly_experienced_resident": "High BMI + Inexperienced Resident",
            "high_bmi_and_no_resident": "High BMI + No Resident",
            "high_bmi_and_highly_experienced_anesthesiologist": "High BMI + Experienced Anesthesiologist",
            "high_bmi_and_scoliosis": "High BMI + Scoliosis",
            "scoliosis_and_highly_experienced_resident": "Scoliosis + Experienced Resident",
            "scoliosis_and_lowly_experienced_resident": "Scoliosis + Inexperienced Resident",
            "scoliosis_and_no_resident": "Scoliosis + No Resident",
            "scoliosis_and_highly_experienced_anesthesiologist": "Scoliosis + Experienced Anesthesiologist",
            "has_scoliosis": "Scoliosis Present",
            "has_dorsalgia": "Dorsalgia Present",
            "has_back_problems": "Back Problems Present",
            "multiple_gestation": "Multiple Gestation",
            "CS_hx": "History of Cesarean Section",
            "high_risk_current_pregnancy": "High Risk Pregnancy (Current)",
            "high_risk_hx": "History of High Risk Pregnancy",
            "iufd": "Intrauterine Fetal Demise (IUFD)",
            "composite_psychosocial_problems": "Psychosocial Problems (Composite)",
            "only_private_insurance": "Private Insurance Only",
            "maternal_language_english": "Maternal Primary Language: English",
            "marital_status_married_or_partner": "Married or Partnered",
            "country_of_origin_USA": "Country of Origin: USA",
            "employment_status_fulltime": "Full-Time Employment",
            "composite_SES_advantage": "Socioeconomic Advantage (Composite)",
            "paresthesias_present": "Paresthesias Present",
            "labor_induction": "Labor Induction",
            "position_posterior_or_transverse": "Posterior or Transverse Presentation",
            "presentation_cephalic": "Cephalic Presentation",

            "anesthesiologist_experience_category = high": "Anesthesiologist Experience Category: High",
            "anesthesiologist_experience_category = low": "Anesthesiologist Experience Category: Low",
            "anesthesiologist_experience_category = moderate": "Anesthesiologist Experience Category: Moderate",
            "anesthesiologist_experience_category = no_anesthesiologist": "Anesthesiologist Experience Category: None",

            "delivery_site = bwh": "Delivery Site: BWH",
            "delivery_site = cdh": "Delivery Site: CDH",
            "delivery_site = mgh": "Delivery Site: MGH",
            "delivery_site = mvh": "Delivery Site: MVH",
            "delivery_site = nch": "Delivery Site: NCH",
            "delivery_site = nwh": "Delivery Site: NWH",
            "delivery_site = slm": "Delivery Site: SLM",
            "delivery_site = wdh": "Delivery Site: WDH",
            "delivery_site_is_bwh": "Delivery Site is BWH",

            "epidural_needle_type = other": "Epidural Needle Type: Other",
            "epidural_needle_type = tuohy": "Epidural Needle Type: Tuohy",
            "epidural_needle_type = weiss": "Epidural Needle Type: Weiss",

            "fetal_position = anterior": "Fetal Position: Anterior",
            "fetal_position = nan": "Fetal Position: Unknown",
            "fetal_position = posterior": "Fetal Position: Posterior",
            "fetal_position = transverse": "Fetal Position: Transverse",
            "fetal_position_is_posterior_or_transverse": "Fetal Position is Posterior or Transverse",

            "fetal_presentation = breech": "Fetal Presentation: Breech",
            "fetal_presentation = cephalic": "Fetal Presentation: Cephalic",
            "fetal_presentation = compound": "Fetal Presentation: Compound",
            "fetal_presentation = lie": "Fetal Presentation: Lie",
            "fetal_presentation = nan": "Fetal Presentation: Unknown",
            "fetal_presentation_is_cephalic": "Fetal Presentation is Cephalic",

            "maternal_ethnicity = Hispanic": "Ethnicity: Hispanic",
            "maternal_ethnicity = Non-Hispanic": "Ethnicity: Non-Hispanic",
            "maternal_ethnicity = Unknown": "Ethnicity: Unknown",

            "maternal_race = Asian": "Race: Asian",
            "maternal_race = Black": "Race: Black",
            "maternal_race = Other/Unknown": "Race: Other/Unknown",
            "maternal_race = White": "Race: White",

            "predicted_lor_depth": "Predicted LOR Depth",

            "resident_experience_category = high": "Resident Experience Category: High",
            "resident_experience_category = low": "Resident Experience Category: Low",
            "resident_experience_category = no_resident": "Resident Experience Category: None",

            "unexpected_delta_lor": "Unexpected delta-LOR",
            "unexpected_delta_lor_squared": "Unexpected delta-LOR squared",

            "true_procedure_type_incl_dpe = cse": "Combined spinal epidural (CSE)",
            "true_procedure_type_incl_dpe = dpe": "Dural puncture epidural (DPE)",
            "true_procedure_type_incl_dpe = epidural": "Conventional epidural",
            "true_procedure_type_incl_dpe = intrathecal": "Intrathecal catheter",
        }
        if name in manual_map:
            return manual_map[name]
        else:
            return name
        
    def clean_pvalue(pval):
        """
        Convert p-values to a consistent format:
        - If p < 0.001, return "< 0.001"
        - Otherwise, return the p-value formatted to 3 decimal places
        """
        if pd.isna(pval):
            return np.nan
        elif pval < 0.001:
            return "< 0.001"
        else:
            return f"{pval:.2g}"

    # Step 2: Apply cleaning
    df["Cleaned Name"] = df["Variable"].apply(clean_variable_name)
    df["pvalue"] = df["pvalue"].apply(clean_pvalue)

    # Step 3: Define thematic categories
    categories = {
        "Maternal Characteristics": [
            "Maternal Age", "BMI Before Pregnancy", "BMI at End of Pregnancy", "Maternal Weight at End of Pregnancy",
        ],
        "Obstetric Info": [
            "Gestational Age (weeks)", "Maternal Age", "Gravidity", "Parity"
        ],
        "Race/Ethnicity": [
            "Race: Asian", "Race: Black", "Race: Other/Unknown", "Race: White",
            "Ethnicity: Hispanic", "Ethnicity: Non-Hispanic", "Ethnicity: Unknown"
        ],
        "Psychosocial/Socioeconomic": [
            "Psychosocial Problems (Composite)", "Socioeconomic Advantage (Composite)", "Private Insurance Only",
            "Full-Time Employment", "Married or Partnered", "Country of Origin: USA",
            "Maternal Primary Language: English"
        ],
        "Comorbidities": [
            "Scoliosis Present", "Dorsalgia Present", "Back Problems Present", "BMI > 40", "Max OB CMI Score",
            "History of Cesarean Section", "High Risk Pregnancy (Current)", "History of High Risk Pregnancy"
        ],
        "Labor and Delivery Characteristics": [
            "Baby Weight", "ROM to Delivery (hours)", "Placement to Delivery (hours)", "ROM to Placement (hours)",
            "Labor Induction", "Posterior or Transverse Presentation", "Cephalic Presentation", "Multiple Gestation",
            "Intrauterine Fetal Demise (IUFD)", "Fetal Presentation: Breech", "Fetal Presentation: Cephalic",
            "Fetal Presentation: Compound", "Fetal Presentation: Lie", "Fetal Presentation: Unknown",
            "Fetal Position: Anterior", "Fetal Position: Unknown", "Fetal Position: Posterior",
            "Fetal Position: Transverse", "Fetal Position is Posterior or Transverse", "Fetal Presentation is Cephalic",
            "Max Prior Pain Score",
        ],
        "Procedure Details": [
            "Loss of Resistance Depth", "Predicted LOR Depth", "Unexpected delta-LOR",
            "Unexpected delta-LOR squared",
            "Number of Neuraxial Attempts", "Neuraxial Catheter Present",
            "Failed Catheter", "Subsequent Neuraxial Catheter", "Subsequent Spinal Procedure",
            "Subsequent Airway Procedure", "Epidural Needle Type: Other", "Epidural Needle Type: Tuohy",
            "Epidural Needle Type: Weiss", "Paresthesias Present", "Combined spinal epidural (CSE)",
            "Dural puncture epidural (DPE)", "Conventional epidural", "Intrathecal catheter",
        ],
        "Delivery site": [
            "Delivery Site: BWH", "Delivery Site: CDH", "Delivery Site: MGH", "Delivery Site: MVH",
            "Delivery Site: NCH", "Delivery Site: NWH", "Delivery Site: SLM", "Delivery Site: WDH",
            "Delivery Site is BWH"
        ],
        "Provider Characteristics": [
            "Resident Present", "Anesthesiologist Present", "Resident Experience Category: High",
            "Resident Experience Category: Low", "Resident Experience Category: None",
            "Anesthesiologist Experience Category: High", "Anesthesiologist Experience Category: Low",
            "Anesthesiologist Experience Category: Moderate", "Anesthesiologist Experience Category: None",
            "Resident Catheter Count (Current)", "Team Catheter Count (Total)",
            "Anesthesiologist Catheter Count (Current)"
        ],
        "Prior Catheters": [
            "Prior Failed Catheters (This Encounter)", "Prior Failed Catheters (Previous Encounter)",
            "All Catheters (All Encounters)"
        ],
        "Feature Interactions": [
            "High BMI + Experienced Resident", "High BMI + Inexperienced Resident", "High BMI + No Resident",
            "High BMI + Experienced Anesthesiologist", "High BMI + Scoliosis", "Scoliosis + Experienced Resident",
            "Scoliosis + Inexperienced Resident", "Scoliosis + Experienced Anesthesiologist",
            "Scoliosis + No Resident"
        ]
    }

    # Step 4: Assign category
    def assign_category(cleaned_name):
        for category, names in categories.items():
            if cleaned_name in names:
                return category
        return "Uncategorized"

    df["Category"] = df["Cleaned Name"].apply(assign_category)

    # Step 5: Sort and show
    df = df.sort_values(by=["Category", "Cleaned Name"])
    df = df[["Category", "Cleaned Name", "Successful Catheters", "Failed Catheters", "All Catheters","pvalue"]]
    df.reset_index(drop=True, inplace=True)

    category_order = [
    "Race/Ethnicity",
    "Delivery site",
    "Maternal Characteristics",
    "Obstetric Info",
    "Labor and Delivery Characteristics",
    "Comorbidities",
    "Psychosocial/Socioeconomic",
    "Procedure Details",
    "Provider Characteristics",
    "Prior Catheters",
    "Feature Interactions",
    "Uncategorized"
]

    missing_categories = set(df['Category'].dropna().unique()) - set(category_order)
    if missing_categories:
        raise ValueError(f"The following categories are present in df['Category'] but missing from category_order: {missing_categories}")

    df['Category'] = pd.Categorical(df['Category'], categories=category_order, ordered=True)
    df = df.sort_values(by=['Category', 'Cleaned Name']).reset_index(drop=True)
    return df



import pandas as pd
from scipy import stats

def univariate_analysis(
        df: pd.DataFrame,
        outcome_col: str = "failed_catheter",
        id_cols: tuple | list = ("anes_procedure_encounter_id_2273", "unique_pt_id"),
        min_expected: float = 5
    ) -> pd.DataFrame:
    """
    χ² / Fisher exact for categorical predictors, Welch t-test for numeric predictors.

    * Categorical variables with >2 levels are exploded into one-vs-rest contrasts.
    * Two-level categorical variables are analysed once as a 2×2 table.
    * Numeric variables use Welch's t-test.

    Returns a tidy DataFrame sorted by p-value.
    """
    if outcome_col not in df.columns:
        raise KeyError(f"Outcome column '{outcome_col}' not found in df.")

    # ---------------------------------------------
    # 1. Identify categorical & numeric predictors
    # ---------------------------------------------
    categorical_cols, numeric_cols = [], []
    for col in df.columns:
        if col in id_cols or col == outcome_col:
            continue
        if df[col].dtype in ("object", "int64", "bool"):
            categorical_cols.append(col)
        elif df[col].dtype == "float64":
            numeric_cols.append(col)

    results = []

    # ---------------------------------------------
    # 2. Categorical variables
    # ---------------------------------------------
    for col in categorical_cols:
        levels = df[col].dropna().unique()

        # ---------- (a) Two-level factor: single 2×2 test ----------
        if len(levels) == 2:
            table = pd.crosstab(df[col], df[outcome_col]).reindex(
                index=levels, columns=[0, 1], fill_value=0
            )
            if (table.values < min_expected).any():
                _, p = stats.fisher_exact(table)
                test_name, stat, dof = "Fisher exact", None, 1
            else:
                stat, p, dof, _ = stats.chi2_contingency(table, correction=False)
                test_name = "chi-square"

            results.append(
                {"Variable": col,
                 "test": test_name,
                 "statistic": stat,
                 "dof": dof,
                 "pvalue": p}
            )

        # ---------- (b) >2 levels: explode into one-vs-rest ----------
        else:
            for level in levels:
                mask = df[col] == level
                table = pd.crosstab(mask, df[outcome_col]).reindex(
                    index=[False, True], columns=[0, 1], fill_value=0
                )
                if (table.values < min_expected).any():
                    _, p = stats.fisher_exact(table)
                    test_name, stat, dof = "Fisher exact", None, 1
                else:
                    stat, p, dof, _ = stats.chi2_contingency(
                        table, correction=False
                    )
                    test_name = "chi-square"

                results.append(
                    {"Variable": f"{col} = {level}",   # <-- spaces around =
                     "test": test_name,
                     "statistic": stat,
                     "dof": dof,
                     "pvalue": p}
                )

    # ---------------------------------------------
    # 3. Numeric variables → Welch’s t-test
    # ---------------------------------------------
    for col in numeric_cols:
        grp0 = df.loc[df[outcome_col] == 0, col].dropna()
        grp1 = df.loc[df[outcome_col] == 1, col].dropna()
        if len(grp0) > 1 and len(grp1) > 1:
            t, p = stats.ttest_ind(grp0, grp1, equal_var=False, nan_policy="omit")
            dof = len(grp0) + len(grp1) - 2
            results.append(
                {"Variable": col,
                 "test": "Welch t-test",
                 "statistic": t,
                 "dof": dof,
                 "pvalue": p}
            )

    # ---------------------------------------------
    # 4. Assemble tidy results
    # ---------------------------------------------
    return (pd.DataFrame(results)
              .sort_values("pvalue")
              .reset_index(drop=True))

