import pandas as pd
import modules.regex_utils as regex_utils

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

