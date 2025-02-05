import pandas as pd
import modules.regex_utils as regex_utils

def print_encounter(df, encounter_id):
    columns = [
    'epic_pmrn',
    "best_timestamp",
    "delivery_datetime",
    "anes_procedure_type_2253",
    "failed_catheter",
    "true_procedure_type",
    "NotePurposeDSC",
    "Regulated_Anesthesiologist_Name",
    "Regulated_Resident_Name",
    "anes_procedure_encounter_id_2273",
    "anes_procedure_note_id_2260",
    "near_duplicate_note_ids",
    "is_worse_near_duplicate",
    "subsequent_proof_of_failure_note_id",
    ]
    existing_cols = [col for col in columns if col in df.columns]
    print(df.loc[df['anes_procedure_encounter_id_2273'] == encounter_id, existing_cols])
    print()
    print()

def explode_separated_procedure_notes(
    df: pd.DataFrame,
    anes_procedure_cols: list,
    delimiter: str = '|'
) -> pd.DataFrame:
    """
    Expands specified anesthesia procedure columns by splitting their string values
    using the given delimiter and exploding them into separate rows.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        delimiter (str): The delimiter used to separate values in the columns.
        anes_procedure_cols (list): List of column names to be split and exploded.

    Returns:
        pd.DataFrame: The exploded DataFrame with each procedure entry in separate rows.

    Raises:
        ValueError: If any of the specified columns are missing from the DataFrame.
        ValueError: If the number of elements after splitting is inconsistent across columns.
    """
    # Validate that all specified columns exist
    missing_cols = set(anes_procedure_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"The following columns are missing from the DataFrame: {missing_cols}")

    # Fill NaN with empty strings to avoid issues during splitting
    df[anes_procedure_cols] = df[anes_procedure_cols].fillna('')

    # Split each specified column by the delimiter into lists
    for col in anes_procedure_cols:
        df[col] = df[col].str.split(delimiter)

    # Validate that all lists in the specified columns have the same length per row
    lengths = df[anes_procedure_cols].map(len)
    if not (lengths.nunique(axis=1) == 1).all():
        inconsistent_rows = lengths[lengths.nunique(axis=1) != 1]
        raise ValueError(f"Inconsistent number of elements across columns in rows: {inconsistent_rows.index.tolist()}")

    # Explode the DataFrame
    df = df.explode(anes_procedure_cols)

    # Reset the index
    df = df.reset_index(drop=True)

    return df

def add_raw_info(df: pd.DataFrame, raw_info_fpath, processed_note_id_col: str, raw_info_cols: list, raw_note_id_col = 'NoteID') -> pd.DataFrame:
    """
    Adds raw information columns to the given DataFrame from a raw info file.
    Matches the processed note IDs with the raw note IDs to retrieve the corresponding raw information.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        raw_info_fpath (str): The file path to the raw info data.
        raw_info_cols (list): List of column names to be added from the raw info data.

    Returns:
        pd.DataFrame: The updated DataFrame with raw information columns added.
    """
    # Load the raw info data
    raw_identified_data = pd.read_csv(raw_info_fpath).set_index(raw_note_id_col)

    for col in raw_info_cols:
        raw_info_dict = raw_identified_data[col].to_dict()
        df[col] = df[processed_note_id_col].map(raw_info_dict)

    return df


def regex_note_text(df: pd.DataFrame, desired_col: str, note_text_col: str = 'NoteTXT', regex_func = None) -> pd.DataFrame:
    """
    Applies a regular expression to the note text column of the DataFrame to extract the desired information.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        desired_col (str): The name of the column to store the extracted information.
        note_text_col (str): The name of the column containing note text.

    Returns:
        pd.DataFrame: The updated DataFrame with the extracted information.
    """

    if desired_col == 'number_of_neuraxial_attempts':
        regex_func = regex_utils.get_number_of_neuraxial_attempts
    
    df[desired_col] = df[note_text_col].apply(regex_func)
    return df


def adjust_date_based_on_dst(timestamp):
    """
    Function to adjust by one day if time is before 0400 (DST) or 0500 (ST)
    """
    # Adjust cutoff time based on DST or standard time
    cutoff_time = pd.Timestamp('04:00:00').time() if timestamp.tz_convert('US/Eastern').dst() != pd.Timedelta(0) else pd.Timestamp('05:00:00').time()
    # Add 24 hours if the time is earlier than the cutoff
    return timestamp + pd.Timedelta(hours=24) if timestamp.time() < cutoff_time else timestamp

def strip_tz_from_col(times: pd.Series):
    """
    Function to strip text like '-0500' from the end of each string in a Series
    """
    return times.str.replace(r'[+-]\d{2}:*\d{2}$', '+0000', regex=True)

def fix_delivery_datetime(df):
    delivery_date = df['delivery_date']
    delivery_time_stripped = strip_tz_from_col(df['delivery_time'])
    delivery_datetime_utc_unadjusted_for_dst = pd.to_datetime(delivery_date + ' ' + delivery_time_stripped,utc=True)
    delivery_datetime_utc_adjusted_for_dst = delivery_datetime_utc_unadjusted_for_dst.apply(adjust_date_based_on_dst)
    df['delivery_datetime'] = delivery_datetime_utc_adjusted_for_dst
    return df

def add_delivery_datetime(df):
    delivery_date = df['delivery_date']
    delivery_time = df['delivery_time']
    df['delivery_datetime'] = pd.to_datetime(delivery_date + ' ' + delivery_time,utc=True)
    return df

def fix_procedure_dos_datetime(df):
    dos_dts_tz_stripped = strip_tz_from_col(df['anes_procedure_dos_dts_2261'])
    dos_dts = pd.to_datetime(dos_dts_tz_stripped)
    df['dos_dts'] = dos_dts
    return df

def label_and_drop_worse_versions_of_duplicates(df: pd.DataFrame, anes_procedure_cols: list, minute_offset = 0, drop=True):
    """
    Go through each encounter to label and delete duplicates
    Duplicates are defined as notes with the same procedure type that are within a certain minute_offset
    """
    minimal_df = df[['anes_procedure_encounter_id_2273','anes_procedure_note_id_2260','anes_procedure_type_2253','best_timestamp']]
    minimal_df.loc[:,'has_near_duplicate'] = 0
    minimal_df.loc[:,'near_duplicate_note_ids'] = None
    minimal_df.loc[:,'time_gap'] = None
    minimal_df = minimal_df.groupby('anes_procedure_encounter_id_2273').apply(lambda x: label_near_duplicate_notes(x, minute_offset = minute_offset), include_groups = False)
    minimal_df = minimal_df.reset_index('anes_procedure_encounter_id_2273')
    minimal_df.loc[:,'blank_anes_procedure_element_col_counts'] = df[anes_procedure_cols].isnull().sum(axis=1)
    minimal_df.loc[:,'is_worse_near_duplicate'] = minimal_df['has_near_duplicate']
    minimal_df = minimal_df.groupby('near_duplicate_note_ids').apply(label_worse_near_duplicates, include_groups = False)
    minimal_df = minimal_df.reset_index('near_duplicate_note_ids')
    if drop:
        minimal_df = minimal_df.loc[minimal_df['is_worse_near_duplicate'] == 0, :]
    return inner_merge(df, minimal_df)

def inner_merge(df1, df2):
    new_cols = [c for c in df2.columns if c not in df1.columns]
    return pd.merge(df1, df2[new_cols], left_index=True, right_index=True, how='inner')

def check_if_near_duplicate(row1, row2, compare_cols, minute_offset):
    """
    Compare two rows and return True if their timestamps are within minute_offset
    and their compare_cols match
    """
    for col in compare_cols:
        if not pd.isnull(row1[col]) and not pd.isnull(row2[col]):
            if row1[col] != row2[col]:
                return False
    if abs(row1['best_timestamp'] - row2['best_timestamp']) > pd.Timedelta(minutes=minute_offset):
        return False
    return True


def label_near_duplicate_notes(encounter, minute_offset = 0):
    """
    Label near_duplicate notes within an encounter using the check_if_near_duplicate function
    """
    indices = encounter.index.tolist()
    for i in range(len(indices)):
        base_idx = indices[i]
        base_row = encounter.loc[base_idx]
        has_near_duplicate = 0
        near_duplicates = [base_row['anes_procedure_note_id_2260']]
        time_gap = []

        for j in range(len(indices)):
            if i == j:
                continue # don't identify self-duplicates
            compare_idx = indices[j]
            compare_row = encounter.loc[compare_idx]


            if check_if_near_duplicate(base_row, compare_row, compare_cols=['anes_procedure_type_2253'], minute_offset = minute_offset):
                has_near_duplicate = 1
                near_duplicates.append(compare_row['anes_procedure_note_id_2260'])
                time_gap = abs(compare_row['best_timestamp'] - base_row['best_timestamp'])

        encounter.at[base_idx, 'has_near_duplicate'] = has_near_duplicate
        encounter.at[base_idx, 'near_duplicate_note_ids'] = str(sorted(near_duplicates))
        encounter.at[base_idx, 'time_gap'] = str(time_gap)

    return encounter

def label_worse_near_duplicates(near_duplicate_set):
    """
    Label the worse near duplicates within a group by setting the 'is_worse_near_duplicate' flag.
    """
    near_duplicate_set.at[near_duplicate_set['blank_anes_procedure_element_col_counts'].idxmin(), 'is_worse_near_duplicate'] = 0
    return near_duplicate_set