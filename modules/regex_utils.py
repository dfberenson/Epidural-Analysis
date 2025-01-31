import re
import numpy as np
import pandas as pd

def get_numeric_structured_info_from_full_note(regex, note_text):
  # Use RegEx to capture the requested structured data
  # If no matches, return NaN
  # If multiple matches, will ensure they are all equal, else return NaN
  matches = re.findall(regex, note_text)
  if len(matches) == 0:
    return np.nan

  if len(matches) == 1:
    return float(matches[0])

  if len(matches) > 1:
    match_list = []
    for match in matches:
      match_list.append(match)

    if all(x == match_list[0] for x in match_list):
      return float(match_list[0])
    else:
      return np.nan



def get_number_of_neuraxial_attempts(note_text):
  if pd.isnull(note_text):
    return np.nan
  return get_numeric_structured_info_from_full_note('Number of attempts: (\\d+)', note_text)
  # Note: the CSE template does not include this info