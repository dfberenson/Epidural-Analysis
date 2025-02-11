import numpy as np
import pandas as pd
import modules.regex_utils as regex_utils
import random

def print_true_mrn(raw_df, encounter_id):
    print(raw_df.loc[raw_df['PatientEncounterID'] == encounter_id, ['epic_pmrn','DateOfServiceDTS']].iloc[0])
    print()

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

def process_secret_CSEs(df, minute_offset = 5):
    """
    Go through each encounter to label and delete secret CSEs
    Secret CSEs are defined as an epidural and a spinal within a certain minute_offset
    The spinal is deleted from each pair
    """
    minimal_df = df[['anes_procedure_encounter_id_2273','anes_procedure_note_id_2260','anes_procedure_type_2253','best_timestamp']]
    minimal_df.loc[:,'is_secret_CSE'] = 0
    minimal_df.loc[:,'secret_CSE_note_ids'] = None
    minimal_df = minimal_df.groupby('anes_procedure_encounter_id_2273').apply(lambda x: label_secret_CSE_notes(x, minute_offset = minute_offset), include_groups = False)
    minimal_df = minimal_df.reset_index('anes_procedure_encounter_id_2273')
    minimal_df = minimal_df.loc[~((minimal_df['is_secret_CSE'] == 1) & (minimal_df['anes_procedure_type_2253'] == 'spinal')), :]
    return inner_merge(df, minimal_df)

def check_if_secret_CSE(row1, row2, minute_offset):
  """
    Compare two rows and return True if exactly one is an epidural, exactly one is a spinal,
    and if their timestamps are within minute_offset
  """
  if abs(row1['best_timestamp'] - row2['best_timestamp']) < pd.Timedelta(minutes=minute_offset):
    if row1['anes_procedure_type_2253'] == 'epidural/intrathecal' or row1['anes_procedure_type_2253'] == 'epidural':
      if row2['anes_procedure_type_2253'] == 'spinal':
        return True
    if row2['anes_procedure_type_2253'] == 'epidural/intrathecal' or row2['anes_procedure_type_2253'] == 'epidural':
      if row1['anes_procedure_type_2253'] == 'spinal':
        return True
  return False

def label_secret_CSE_notes(encounter, minute_offset = 5):
  """
  Label secret CSE notes within an encounter using the check_if_secret_CSE function
  """

  indices = encounter.index.tolist()

  for i in range(len(indices)):
    base_idx = indices[i]
    base_row = encounter.loc[base_idx]
    is_secret_CSE = 0
    secret_CSEs = [base_row['anes_procedure_note_id_2260']]

    for j in range(len(indices)):
      if i == j:
        continue # don't identify self-duplicates
      compare_idx = indices[j]
      compare_row = encounter.loc[compare_idx]

      if check_if_secret_CSE(base_row, compare_row, minute_offset = minute_offset):
        is_secret_CSE = 1
        secret_CSEs.append(compare_row['anes_procedure_note_id_2260'])

    encounter.at[base_idx, 'is_secret_CSE'] = is_secret_CSE
    encounter.at[base_idx, 'secret_CSE_note_ids'] = str(sorted(secret_CSEs))

  return encounter

def classify_true_procedure_type(df: pd.DataFrame, intelligent=False):
    if intelligent:
        # TODO: Implement intelligent classification based on aspiration
        raise NotImplementedError("Intelligent classification not yet implemented")
    df['is_intrathecal_catheter'] = (df['anes_procedure_type_2253'] == 'intrathecal').astype(int)
    df['true_procedure_type'] = np.where(
        df['is_secret_CSE'] == True,'cse',
        df['anes_procedure_type_2253'])
    df.loc[
        (df['true_procedure_type'].isin(['epidural/intrathecal', 'intrathecal'])) &
        (df['is_intrathecal_catheter'] == True),
        'true_procedure_type'] = 'intrathecal'
    df.loc[
        (df['true_procedure_type'] == 'epidural/intrathecal') &
        (df['is_intrathecal_catheter'] == False),
        'true_procedure_type'] = 'epidural'
    df['is_neuraxial_catheter'] = (df['true_procedure_type'].isin(['cse', 'epidural', 'intrathecal'])).astype(int)
    df['is_spinal'] = (df['true_procedure_type'] == 'spinal').astype(int)
    df['is_airway'] = (df['true_procedure_type'] == 'airway').astype(int)
    return df


def classify_encounter_failures(encounter):
    """
    Classify neuraxial catheter failures within an encounter.
    A neuraxial catheter failure is defined as the presence of a neuraxial catheter procedure
    followed by a subsequent neuraxial catheter, spinal, or airway procedure within the same encounter.
    If the index procedure is not a neuraxial catheter, it will be labeled 0
    """

    # Identify rows where 'is_neuraxial_catheter' == 1
    neuraxial_rows = encounter[encounter['is_neuraxial_catheter'] == 1]

    # If no neuraxial catheter procedures, return encounter as is
    if neuraxial_rows.empty:
        return encounter

    # Create a mask for failure-defining events within the encounter
    # Failure-defining events are neuraxial catheters, spinals, and airways
    failure_defining_event_mask = encounter[['is_neuraxial_catheter','is_spinal','is_airway']].any(axis=1)

    # Get the indices of events
    failure_defining_event_indices = encounter.index[failure_defining_event_mask]

    # Iterate over neuraxial catheter rows
    for idx in neuraxial_rows.index:
        current_time = encounter.at[idx, 'best_timestamp']

        # Find subsequent events
        # This relies on correct ordering by best_timestamp
        subsequent_failure_defining_events = encounter.loc[failure_defining_event_indices]
        subsequent_failure_defining_events = subsequent_failure_defining_events[subsequent_failure_defining_events['best_timestamp'] > current_time]

        # Initialize flags
        has_subsequent_neuraxial_catheter = 0
        has_subsequent_spinal = 0
        has_subsequent_airway = 0
        failed_catheter = 0
        subsequent_proof_of_failure_note_id = None

        # Check for subsequent procedures
        if not subsequent_failure_defining_events.empty:
            # Update flags based on any occurrence in subsequent events
            has_subsequent_neuraxial_catheter = int((subsequent_failure_defining_events['is_neuraxial_catheter'] == 1).any())
            has_subsequent_spinal = int((subsequent_failure_defining_events['is_spinal'] == 1).any())
            has_subsequent_airway = int((subsequent_failure_defining_events['is_airway'] == 1).any())
            failed_catheter = int(has_subsequent_neuraxial_catheter or has_subsequent_spinal or has_subsequent_airway)
            subsequent_proof_of_failure_note_id = subsequent_failure_defining_events['anes_procedure_note_id_2260'].tolist()

            encounter.at[idx, 'has_subsequent_neuraxial_catheter'] = has_subsequent_neuraxial_catheter
            encounter.at[idx, 'has_subsequent_spinal'] = has_subsequent_spinal
            encounter.at[idx, 'has_subsequent_airway'] = has_subsequent_airway
            encounter.at[idx, 'failed_catheter'] = failed_catheter
            encounter.at[idx, 'subsequent_proof_of_failure_note_id'] = str(subsequent_proof_of_failure_note_id)

    return encounter

def label_failed_catheters(df):
    """
    Classify neuraxial catheter failures in the dataframe.
    A neuraxial catheter failure is defined as the presence of a neuraxial catheter procedure
    followed by a subsequent neuraxial catheter, spinal, or airway procedure within the same encounter.
    If the index procedure is not a neuraxial catheter, it will be labeled 0.
    """
    minimal_df = df[['anes_procedure_encounter_id_2273','anes_procedure_note_id_2260','best_timestamp','is_neuraxial_catheter','is_spinal','is_airway']]
    minimal_df['has_subsequent_neuraxial_catheter'] = 0
    minimal_df['has_subsequent_spinal'] = 0
    minimal_df['has_subsequent_airway'] = 0
    minimal_df['failed_catheter'] = 0
    minimal_df['subsequent_proof_of_failure_note_id'] = None

    minimal_df = minimal_df.groupby('anes_procedure_encounter_id_2273').apply(classify_encounter_failures, include_groups = False)
    minimal_df = minimal_df.reset_index('anes_procedure_encounter_id_2273')
    return inner_merge(df, minimal_df)

def count_prior_catheters(df):
    df = df.sort_values(by='best_timestamp', ascending=True)
    df = df.groupby('anes_procedure_encounter_id_2273').apply(count_prior_failed_catheters_this_enc, include_groups = False)
    df = df.reset_index('anes_procedure_encounter_id_2273')
    df = df.groupby('epic_pmrn').apply(count_prior_failed_catheters_all_enc, include_groups = False)
    df = df.reset_index('epic_pmrn')
    df = df.groupby('epic_pmrn').apply(count_prior_all_catheters_all_enc, include_groups = False)
    df = df.reset_index('epic_pmrn')
    df['prior_failed_catheters_prev_enc'] = df['prior_failed_catheters_all_enc'] - df['prior_failed_catheters_this_enc']
    return df

def count_prior_failed_catheters_this_enc(group):
    group['prior_failed_catheters_this_enc'] = (group['failed_catheter'].cumsum() - group['failed_catheter']).astype(float)
    return group

def count_prior_failed_catheters_all_enc(group):
    group['prior_failed_catheters_all_enc'] = (group['failed_catheter'].cumsum() - group['failed_catheter']).astype(float)
    return group

def count_prior_all_catheters_all_enc(group):
    group['prior_all_catheters_all_enc'] = (group['is_neuraxial_catheter'].cumsum() - group['is_neuraxial_catheter']).astype(float)
    return group

def prior_catheter_helper(group, type):
    """
    Helper function to count the number of prior catheters of a certain type
    """
    match type:
        case 'failed_this_encounter':
            group['prior_failed_catheters_this_enc'] = (group['failed_catheter'].cumsum() - group['failed_catheter']).astype(float)
        case 'all_enc':
            group['prior_failed_catheters_all_enc'] = (group['failed_catheter'].cumsum() - group['failed_catheter']).astype(float)
            group['prior_all_catheters_all_enc'] = (group['is_neuraxial_catheter'].cumsum() - group['is_neuraxial_catheter']).astype(float)
    return group


def get_vals_prior_to_timestamp(row, vals_col, times_col, best_timestamp_col="best_timestamp"):
    """
    Extract all pain scores that have timestamp < row[best_timestamp_col].

    row: a single row of your DataFrame (a pd.Series)
    best_timestamp_col: name of the column in your DataFrame that contains
                       the 'best_timestamp' to compare against

    Returns a list of 'prior' scores or NaN if none exist.
    """
    # Extract the raw strings
    times_str = row[times_col]
    vals_str = row[vals_col]

    # If either is missing, return NaN
    if pd.isna(times_str) or pd.isna(vals_str):
        return np.nan

    # Convert to lists
    times_list = times_str.split("|")
    vals_list = vals_str.split("|")

    # Safely convert both times and best_timestamp to datetime
    try:
        times_dt = pd.to_datetime(times_list,utc=True,format='mixed')
        # This assumes your row also has a column called best_timestamp_col
        best_dt = pd.to_datetime(row[best_timestamp_col])
    except:
        # If conversion fails, return NaN
        return np.nan

    # Filter out all scores whose timestamp is strictly less than best_timestamp
    prior_vals = []
    for t, v in zip(times_dt, vals_list):
        if t < best_dt:
            prior_vals.append(float(v))

    # If no scores remain, return NaN, else return them joined or as list
    return prior_vals if prior_vals else np.nan

def handle_pain_scores(df):
    """
    Extract the list of pain scores that occurred before the best timestamp.
    Returns the maximum of those scores, divided by 10 (since the scores are reported in the data as 0-100).
    """
    df["prior_pain_scores"] = df.apply(get_vals_prior_to_timestamp, vals_col = 'timeseries_intrapartum_pain_score_2242', times_col = 'timeseries_intrapartum_pain_score_datetime_2242', axis=1)
    df["prior_pain_scores_max"] = df["prior_pain_scores"].apply(
    lambda scores: max(map(float, scores)) if isinstance(scores, list) and scores else np.nan) / 10
    return df

def handle_cmi_scores(df):
    """
    Extract the list of CMI scores that occurred before the best timestamp.
    Returns the maximum of those scores.
    """
    df["prior_ob_cmi_scores"] = df.apply(get_vals_prior_to_timestamp, vals_col = 'ob_cmi_2308', times_col = 'ob_cmi_datetime_2308', axis=1)
    df["prior_ob_cmi_scores_max"] = df["prior_ob_cmi_scores"].apply(
    lambda scores: max(map(float, scores)) if isinstance(scores, list) and scores else np.nan)
    return df

def handle_dpe(df):
    df['dpe'] = df['anes_procedure_dpe_2262'] == 'yes'
    df['true_procedure_type_incl_dpe'] = df['true_procedure_type']
    df.loc[df['dpe'] == True, 'true_procedure_type_incl_dpe'] = 'dpe'
    return df

def handle_lor_depth(df):
    """
    Divide LOR by 10 if it is > 20
    # Code to evaluate suspiciously high LORs
    # For these, if we divide LOR by 10, the the catheter is taped around 4-5 cm deeper
    # So most likely these suspiciously high LORs are missing decimal points
    high_LORs = df.sort_values(by='lor_depth',ascending=False).head(100)['lor_depth']
    print(high_LORs.to_list())
    plt.hist(high_LORs)
    print(df.sort_values(by='lor_depth',ascending=False).head(100)['anes_procedure_catheter_depth_2266'].to_list())
    """
    df['lor_depth'] = df['anes_procedure_lor_depth_2265'].replace('', np.nan).astype(float)
    df['lor_depth'] = np.where(df['lor_depth'] > 20, df['lor_depth'] / 10, df['lor_depth'])
    return df

def numerify_columns(df, columns_to_convert):
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    return df

def handle_elapsed_times(df):
    """
    From my analyses, procedures where many days elapse between placement and delivery are NOT labor analgesia procedures. They can be totally unrelated procedures like knee surgery, or obstetrical procedures like ECVs, or (rarely) analgesia for false labor. In the latter case, if labor does not progress and the patient returns to antepartum, the anesthesia encounter will termiante and a new encounter will be used for subsequent labor. In that case, an epidural placed in the second encounter will NOT prove failure of the first since it will have a different encounter_id.

    For these reasons, I eliminate rows where there is more than 7 days between placement and delivery.

    Due to the UTC bug discussed above, a true 1859 EPL followed by 1900 delivery would be translated to 2359 EPL AFTER 0000 delivery (without the delivery_date incrementing appropriately)

    A more thorough algorithm could look at the timing of Anesthesia Stop compared to delivery, and/or confirm that the title of the anesthesia encounter is Labor Epidural or Cesarean Section.
    """
    df['rom_thru_delivery_hours'] = df['secs_rom_thru_delivery_2197'] / 3600
    # If ROM through Delivery is more than 30 days, assume erroneous and make it NaN
    df['rom_thru_delivery_hours'] = np.where(df['rom_thru_delivery_hours'] <= 30*24, df['rom_thru_delivery_hours'],np.nan)
    df['maternal_age_years'] = (df['best_timestamp'] - df['maternal_dob']).dt.days / 365.25
    df['gestational_age_weeks'] = df['gestational_age_2052'] / 7
    df['placement_to_delivery_hours'] = (df['delivery_datetime'] - df['best_timestamp']).dt.total_seconds() / 3600
    df['placement_to_delivery_hours'] = np.where((df['placement_to_delivery_hours'] > -1) & (df['placement_to_delivery_hours'] <= 7*24),
                                             df['placement_to_delivery_hours'], np.nan)
    df['rom_to_placement_hours'] = df['rom_thru_delivery_hours'] - df['placement_to_delivery_hours']
    return df

def handle_anesthesiologists(df):
    """
    Regulate names and count prior catheters
    """
    df['Regulated_Anesthesiologist_Name'] = df['anes_procedure_anesthesiologist_2255'].dropna().apply(regex_utils.regulate_name)
    df['Regulated_Resident_Name'] = df['anes_procedure_resident_2256'].dropna().apply(regex_utils.regulate_name)
    df['Regulated_Anesthesiologist_Name'] = df['Regulated_Anesthesiologist_Name'].replace('', np.nan)
    df['Regulated_Resident_Name'] = df['Regulated_Resident_Name'].replace('', np.nan)
    df = df.sort_values('best_timestamp')
    df['current_anesthesiologist_catheter_count'] = (
        df.groupby('Regulated_Anesthesiologist_Name')['is_neuraxial_catheter']
        .cumsum()
    )
    df['current_resident_catheter_count'] = (
        df.groupby('Regulated_Resident_Name')['is_neuraxial_catheter']
        .cumsum()
    )
    df['highly_experienced_anesthesiologist'] = np.where(df['current_anesthesiologist_catheter_count'] > 500, 'yes',
                                                    np.where(df['current_anesthesiologist_catheter_count'] <= 500, 'no', 'none'))
    df['moderately_experienced_anesthesiologist'] = np.where(df['current_anesthesiologist_catheter_count'] > 100, 'yes',
                                                        np.where(df['current_anesthesiologist_catheter_count'] <= 100, 'no', 'none'))
    df['highly_experienced_resident'] = np.where(df['current_resident_catheter_count'] > 50, 'yes',
                                                    np.where(df['current_resident_catheter_count'] <= 50, 'no', 'none'))
    return df

def engineer_categorical_variables(df):
    df['has_scoliosis'] = df['icd_scoliosis_2053'] == True
    df['has_dorsalgia'] = df['icd_dorsalgia_2104'] == True

    # prompt: create a column "has_back_problems" that is 1 where any of the following are True, else 0. Handle NaN.
    # Define the columns related to back problems
    back_problem_cols = [
        'icd_scoliosis_2053',
        'icd_spinal_fusion_2056',
        'icd_congenital_deformity_spine_2059',
        'icd_ra_and_sctds_2086',
        'icd_kyphosis_and_lordosis_2089',
        'icd_spinal_osteochondrosis_2092',
        'icd_spondylopathies_and_deforming_dorsopathies_2095',
        'icd_intervertebral_disc_disorders_2098',
        'icd_ehlers_minus_danlos_2101',
    ]

    # Note that spondyolopathies_and_deforming_dorsopathies are by far the biggest contributors

    # Create the 'has_back_problems' column
    df['has_back_problems'] = df[back_problem_cols].any(axis=1)

    df['maternal_race'] = np.select(
    [
        df['maternal_race_2111'] == 'White',
        df['maternal_race_2111'] == 'Asian',
        df['maternal_race_2111'] == 'Black'
    ],
    [
        'White',
        'Asian',
        'Black'
    ],
    default='Other/Unknown')
    # Create a new column for maternal ethnicity, using Regex to look at the start of each line to find "Hispanic" or "Not Hispanic" (to avoid concatenations like "Hispanic|Hispanic")
    df['maternal_ethnicity'] = np.where(df['maternal_ethnicity_2112'].str.contains(r'^Hispanic', regex=True), 'Hispanic', np.where(df['maternal_ethnicity_2112'].str.contains(r'^Not Hispanic', regex=True), 'Non-Hispanic', 'Unknown'))
    composite_social_columns = [
    "drug_abuse_during_parent_2144",
    "high_risk_social_problems_parent_2154",
    "high_risk_insufficient_antenatal_care_parent_2157",
    "icd_major_mental_health_disorder_2178",
    "education_problems_2203",
    "employment_problems_2206",
    "adverse_occupational_2209",
    "housing_problems_2212",
    "adjustment_problems_2215",
    "relationship_problems_2218",
    "other_psychosocial_2221",
    "smoker_during_pregnancy_parent_2117",
    "drug_abuse_before_parent_2142",
    "alcohol_during_parent_2147",
    ]
    df['composite_psychosocial_problems'] = df[composite_social_columns].any(axis=1)

    # prompt: create column 'only_private_insurance' for any row where public_insurance_2114 does NOT contains the string "public", ignore case
    # Assuming 'df' is your DataFrame.
    df['only_private_insurance'] = ~df['public_insurance_2114'].str.contains('public', case=False, na=False)

    # prompt: create a column maternal_language_english for any row where maternal_language is english
    df['maternal_language_english'] = df['maternal_language_2113'] == 'english'

    # prompt: create a column marital_status_married_or_partner for any row where marital_status_2184 is 'married' or 'partner'
    df['marital_status_married_or_partner'] = df['marital_status_2184'].apply(lambda x: True if x in ['married', 'partner'] else False)

    # prompt: create a column country_of_origin_USA that is country_of_origin_2186 == united states
    df['country_of_origin_USA'] = df['country_of_origin_2186'] == 'united states'

    # prompt: create a column employment_status_fulltime that is employment_status_2187 == full time
    df['employment_status_fulltime'] = df['employment_status_2187'] == 'full time'

    composite_SES_columns = [
    "only_private_insurance",
    "maternal_language_english",
    "marital_status_married_or_partner",
    "country_of_origin_USA",
    "employment_status_fulltime",
    ]
    df['composite_SES_advantage'] = df[composite_SES_columns].all(axis=1)

    # prompt: create a column epidural_needle_type based on anes_procedure_epidural_needle_2263 that can have values "tuohy","weiss", or "other"
    # Create the 'epidural_needle_type' column based on 'anes_procedure_epidural_needle_2263'
    df['epidural_needle_type'] = df['anes_procedure_epidural_needle_2263'].map({
        'tuohy': 'tuohy',
        'weiss': 'weiss',
    }).fillna('other')

    # prompt: create a column paresthesias_present that is anes_procedure_paresthesias_2270 either "transient" or "yes"

    # Create the 'paresthesias_present' column
    df['paresthesias_present'] = df['anes_procedure_paresthesias_2270'].apply(lambda x: True if x == 'yes' or x == 'transient' else False)

    df['delivery_site'] = np.where(df['delivery_site_2188'] == 'mgb', np.nan, df['delivery_site_2188'])
    df['delivery_site_bwh'] = df['delivery_site'] == 'bwh'
    
    df['labor_induction'] = df[[
    'induction_oxytocin_2189','induction_cervical_balloon_2190','induction_misoprostol_2191','induction_arom_2192','induction_foley_easy_2193']].any(axis=1)

    df['position_posterior_or_transverse'] = (df['fetal_presentation_position_2247'] == 'posterior') | (df['fetal_presentation_position_2247'] == 'transverse')
    df['presentation_cephalic'] = df['fetal_presentation_category_2243'] == 'cephalic'

    return df

def create_unique_id(df):
    # Define identifier range (6-digit numbers)
    id_len = 8
    min_id, max_id = 10**(id_len-1), 10**id_len - 1

    # Create mapping of unique MRNs to unique random identifiers
    unique_mrns = df['epic_pmrn'].unique()
    mapping = dict(zip(unique_mrns, random.sample(range(min_id, max_id+1), len(unique_mrns))))

    # Map to a new column in DataFrame
    df['unique_pt_id'] = df['epic_pmrn'].map(mapping)
    return df