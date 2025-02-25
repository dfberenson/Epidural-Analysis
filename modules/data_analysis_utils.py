import pandas as pd
import re
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def prepend_char_to_most_common(df, char, cols_to_ignore=None):
    """
    For each column in df, if the most common value is a string,
    prepend char to it. Otherwise, do nothing.
    """
    if cols_to_ignore is None:
        cols_to_ignore = []

    for col in df.columns:
        if col in cols_to_ignore:
            continue

        # Get the most common value
        most_common_value = df[col].mode().iloc[0]

        # Check if it's a string
        if isinstance(most_common_value, str):
            # Prepend char to it
            df[col] = df[col].apply(lambda x: char + x if x == most_common_value else x)
    return df


def parse_param_name(param_name):
    """
    Parses a statsmodels parameter name like:
        'C(col)[T.value]'
    and returns the level name 'value'.
    """
    
    # Regex for the typical pattern: C(colName)[T.levelName]
    pattern = r'.*\[T\.(.+)\]'
    match = re.match(pattern, param_name)
    if match:
        level_name = match.group(1)
        return level_name
    # If it doesn't match, assume it's some other type of parameter (e.g., numeric var)
    return ''


def all_regressions_each_dummy(df, outcome_col='failed_catheter'):
    """
    Fits a univariate logistic regression for each column in df (except outcome_col).
    For numeric columns, you get a single slope term.
    For categorical columns, you get one dummy variable per level (minus the reference).
    Then plots x=coefficient, y=-log10(p-value) for *all* those dummy variables.
    """
    

    results = []

    for col in df.columns:
        # Skip the outcome column
        if col == outcome_col:
            continue

        # Skip encounter_id
        if col == "anes_procedure_encounter_id_2273" or col == "unique_pt_id":
            continue
        
        # Skip datetime or other unsupported types
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        # Subset to non-null rows in outcome & predictor
        temp_df = df[[outcome_col, col]].dropna()
        
        # Skip if not enough variation
        if temp_df[col].nunique() < 2 or temp_df[col].count() < 5:
            continue
        
        # Build formula
        # Wrap in C() if categorical
        if pd.api.types.is_numeric_dtype(temp_df[col]):
            formula = f"{outcome_col} ~ {col}"
        else:
            formula = f"{outcome_col} ~ C({col})"
        
        # Fit the logistic model
        try:
            model = smf.logit(formula, data=temp_df).fit(disp=False)
        except Exception as e:
            print(f"Skipping column '{col}' due to fitting error: {e}")
            continue
        
        # For each parameter (except the Intercept),
        # capture the coefficient and p-value.
        for param_name in model.params.index:
            if param_name == 'Intercept':
                continue
            
            coef = model.params.loc[param_name]
            pval = model.pvalues.loc[param_name]
            
            # You might want to create a cleaner label for the parameter.
            # For categorical variables, param_name will look like 'C(col)[T.level]'
            # We'll store the raw param_name, but you can parse it if you like.

            results.append({
                'column': col,
                'param_name': param_name,
                'coef': coef,
                'pval': pval
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("No valid predictors found.")
        return pd.DataFrame()

    # Sort by p-value (optional)
    results_df = results_df.sort_values(by='pval')

    
    return results_df

# Remove digits from the graph annotations
def remove_nums(string):
    """
    Removes numbers from a string.
    """
    return ''.join([i for i in string if not i.isdigit()])