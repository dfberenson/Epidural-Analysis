import warnings
import pandas as pd
import numpy as np
import re
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

FIGURE_FPATH = "C:\\Users\\User\\OneDrive - Mass General Brigham\\Epidural project\\Figures\\"

def reduce_cols(df):
    reduced_cols = [
    'unique_pt_id','anes_procedure_encounter_id_2273',
    'failed_catheter',
    'rom_to_placement_hours',
    'maternal_age_years','parity_2048',
    'multiple_gestation','labor_induction',
    'gestational_age_weeks','baby_weight_2196',
    'fetal_position_is_posterior_or_transverse',
    'fetal_presentation_is_cephalic',
    'bmi_end_pregnancy_2044',
    'delivery_site_is_bwh',
    'anesthesiologist_experience_category','resident_experience_category',
    'high_bmi_and_scoliosis','has_scoliosis','has_dorsalgia','has_back_problems',
    'prior_ob_cmi_scores_max','CS_hx','high_risk_current_pregnancy','high_risk_hx','iufd',
    'composite_psychosocial_problems','composite_SES_advantage',
    'true_procedure_type_incl_dpe',
    'lor_depth','unexpected_delta_lor_squared',
    'prior_pain_scores_max','paresthesias_present','number_of_neuraxial_attempts',
    'prior_failed_catheters_this_enc','prior_failed_catheters_prev_enc','prior_all_catheters_all_enc'
    ]
    return df[reduced_cols]

def prepend_char(df, char, chosen_default_categories=None, cols_to_ignore=None):
    for col, default_val in chosen_default_categories.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: char + x if x == default_val else x)
        else:
            print(f"Warning: {col} not found in DataFrame columns.")

    cols_to_ignore.extend(chosen_default_categories.keys())
    df = prepend_char_to_most_common(df, char, cols_to_ignore=cols_to_ignore)
    return df


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

def save_text(text, fname):
    # Open (or create) a text file in write mode and write the content
    with open(FIGURE_FPATH + fname + ".txt", "w") as file:
        file.write(text)


def show_text(text):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Add text to the center of the plot
    ax.text(0.5, 0.5, text,
            fontsize=24, ha='center', va='center')

    # Remove axes for a cleaner look
    ax.axis('off')

    # Display the figure
    plt.show()

def plot_coef_vs_pval(results_df):
    fig, ax = plt.subplots(figsize=(8, 6))

    offset = 1e-300  # so we don't take log10(0)
    x_vals = results_df[results_df['pval'] < 0.9]['coef']
    y_vals = -np.log10(results_df[results_df['pval'] < 0.9]['pval'] + offset)

    sc = ax.scatter(x_vals, y_vals, color='blue')

    # Annotate each point
    for i, row in results_df[results_df['pval'] < 0.9].iterrows():
        ax.text(
            row['coef'],
            -np.log10(row['pval'] + offset),
            remove_nums(str(row['column'] + ('__' + str(row['category_variable'])) if row['category_variable'] != '' else row['column'])),
            fontsize=8,
            ha='left',
            va='bottom'
        )

    # Add a reference line for p=0.05
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')

    ax.set_xlabel('Coefficient')
    ax.set_ylabel('-log10(p-value)')
    ax.set_title(f'Logistic Regressions for Catheter_Failure ~ Each Predictor (All Dummies)')
    ax.legend()

    plt.tight_layout()
    plt.show()


def preprocess_data(data):
    # Drop columns with more than 80% missing values
    threshold = len(data) * 0.2
    data_cleaned = data.dropna(thresh=threshold, axis=1)

    # Drop rows where target variable is missing
    data_cleaned = data_cleaned.dropna(subset=["failed_catheter"])

    # Separate features and target variable
    X = data_cleaned.drop(columns=["failed_catheter"], errors='ignore')
    y = data_cleaned["failed_catheter"]

    ##############################################################################
    # 1. Extract the group labels and remove them from X if it's just an ID column
    ##############################################################################
    groups = X['unique_pt_id']  # Save group labels
    # If you do NOT want to use `anes_procedure_encounter_id_2273` as a feature:
    X = X.drop(columns=["unique_pt_id","anes_procedure_encounter_id_2273"])  

    ##############################################################################
    # 2. Split using GroupShuffleSplit instead of train_test_split
    ##############################################################################
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Identify numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing into a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Fit and transform the training data
    X_train_preprocessed_array = preprocessor.fit_transform(X_train)
    feature_names = preprocessor.get_feature_names_out()
    X_train_preprocessed = pd.DataFrame(X_train_preprocessed_array, columns=feature_names, index=X_train.index)

    # Transform the test data using the same feature names
    X_test_preprocessed_array = preprocessor.transform(X_test)
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed_array, columns=feature_names, index=X_test.index)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test

def do_logistic_regression(X_train_preprocessed, X_test_preprocessed, y_train, y_test):
    # Train logistic regression with class weights
    logistic_model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', n_jobs=1)
    logistic_model.fit(X_train_preprocessed, y_train)

    # Make predictions
    y_pred = logistic_model.predict(X_test_preprocessed)
    y_pred_prob = logistic_model.predict_proba(X_test_preprocessed)[:, 1]

    # Evaluate the model
    evaluation_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob),
        "classification_report": classification_report(y_test, y_pred)
    }

    # Print evaluation metrics
    print("Model Evaluation:")
    for metric, value in evaluation_metrics.items():
        if metric == "classification_report":
            print("\nClassification Report:\n", value)
        else:
            print(f"{metric.capitalize()}: {value:.4f}")
    
    return logistic_model

def preprocess_data_for_statsmodels(data):
    # Drop columns with more than 80% missing values
    threshold = len(data) * 0.2
    data_cleaned = data.dropna(thresh=threshold, axis=1)
    data_cleaned = data_cleaned.dropna(subset=["failed_catheter"])

    X = data_cleaned.drop(columns=["failed_catheter"], errors='ignore')
    y = data_cleaned["failed_catheter"]

    # 2. Group-aware split
    groups = X['unique_pt_id']
    X = X.drop(columns=["anes_procedure_encounter_id_2273","unique_pt_id"])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()

    # 3. Identify numeric vs. categorical
    numeric_features = X_train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()

    # 4. Impute numeric
    num_imputer = SimpleImputer(strategy='median')
    X_train[numeric_features] = num_imputer.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = num_imputer.transform(X_test[numeric_features])

    # 5. Scale numeric
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])

    # 6. Impute categorical
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical_features] = cat_imputer.fit_transform(X_train[categorical_features])
    X_test[categorical_features] = cat_imputer.transform(X_test[categorical_features])

    # 7. One-hot encode categorical
    X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True,dtype=int).astype(float)
    X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True,dtype=int).astype(float)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    
    return X_train, X_test, y_train, y_test

def do_logistic_regression_with_statsmodels(X_train, X_test, y_train, y_test):
    # Fit logistic regression with Statsmodels
    X_train_const = sm.add_constant(X_train)
    logit_model = sm.Logit(y_train, X_train_const)
    result = logit_model.fit(disp=0)

    # Predict
    X_test_const = sm.add_constant(X_test, has_constant='add')
    y_pred_prob = result.predict(X_test_const)

    # Evaluate precision and recall at different thresholds
    evaluation_metrics_by_threshold = []

    for i in range(0, 21):
        prediction_threshold = i / 20
        y_pred = (y_pred_prob >= prediction_threshold).astype(int)

        # 10. Evaluate
        evaluation_metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred,zero_division=np.nan),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_prob),
            # "classification_report": classification_report(y_test, y_pred)
        }
        evaluation_metrics_by_threshold.append(evaluation_metrics)

    result_summ = result.summary(alpha = 0.001)
    return y_pred_prob, result_summ, evaluation_metrics, evaluation_metrics_by_threshold

def prepare_logit_tables(result_summ):
    logit_result_table = result_summ.tables[0].data

    logit_predictor_table = pd.DataFrame(result_summ.tables[1].data)
    # 1. Set the first row as the new column headers.
    logit_predictor_table.columns = logit_predictor_table.iloc[0]

    # 2. Remove the first row (since it's now serving as header).
    logit_predictor_table = logit_predictor_table[1:]

    # 3. Set the first column (after the column headers update) as the index.
    # Here, `df.columns[0]` represents the name of the first column.
    logit_predictor_table = logit_predictor_table.set_index(logit_predictor_table.columns[0])

    # 4. Sort by the 'P>|z|' column in ascending order.
    logit_predictor_table = logit_predictor_table.sort_values('P>|z|')

    # 5. Calculate the odds ratio (OR) and the 95% confidence intervals
    logit_predictor_table['OR'] = np.exp(logit_predictor_table['coef'].astype(float))
    logit_predictor_table['OR_lower'] = np.exp(logit_predictor_table['[0.0005'].astype(float))
    logit_predictor_table['OR_upper'] = np.exp(logit_predictor_table['0.9995]'].astype(float))

    # 6. Create the 'OR (95% CI)' column
    logit_predictor_table['OR (99.9% CI)'] = logit_predictor_table.apply(
        lambda row: f"{row['OR']:.2f} ({row['OR_lower']:.2f} - {row['OR_upper']:.2f})", axis=1)

    return logit_result_table, logit_predictor_table

RENAME_MAP = {
    'gestational_age_weeks': 'Gestational Age (per week)',
    'rom_to_placement_hours': 'Time from ROM to placement (per hour)',
    'delivery_site_is_bwh': 'Delivery at our obstetric teaching hospital (vs other)',
    'baby_weight_2196': 'Weight of neonate (per kg)',
    'bmi_end_pregnancy_2044': 'BMI (per kg/m^2)',
    'parity_2048': 'Parity (per birth)',
    'has_dorsalgia': 'Back pain (vs none)',
    'has_back_problems': 'Scoliosis or other back problems (vs none)',
    'prior_pain_scores_max': 'Max pain score prior to placement (per unit 0-10)',
    'composite_SES_advantage': 'All socioeconomic advantages (vs not all)',
    'composite_psychosocial_problems': 'Any psychosocial risk factors (vs none)',
    'prior_all_catheters_all_enc': 'Prior catheters across all encounters (per catheter)',
    'prior_failed_catheters_this_enc': 'Prior failed catheters in this encounter (per failure)',
    'prior_failed_catheters_prev_enc': 'Prior failed catheters in prior encounters (per failure)',
    'maternal_age_years': 'Maternal age (per year)',
    'labor_induction': 'Induced labor (vs not)',
    'high_risk_hx': 'History of high risk pregnancy (vs not)',
    'high_risk_current_pregnancy': 'Current high risk pregnancy (vs not)',
    'CS_hx': 'History of Cesarean section (vs not)',
    'iufd': 'Intrauterine fetal demise',
    'multiple_gestation': 'Multiple gestation (vs not)',
    'prior_ob_cmi_scores_max': 'Max OB-CMI score prior to placement (per unit)',
    'high_bmi_and_scoliosis': 'BMI > 40 *and* scoliosis (vs not both)',
    'has_scoliosis': 'Scoliosis (vs none)',
    'fetal_position_is_posterior_or_transverse': 'Posterior or transverse fetal position (vs other)',
    'fetal_presentation_is_cephalic': 'Cephalic fetal presentation (vs other)',
    # procedural factors below
    'lor_depth': 'Depth to loss of resistance (per cm)',
    'unexpected_delta_lor_squared': 'Squared difference between observed LOR and BMI-predicted LOR (per cm^2)',
    'anesthesiologist_experience_category_high': 'Highly experienced attending anesthesiologist (vs moderately experienced)',
    'anesthesiologist_experience_category_low': 'Minimally experienced attending anesthesiologist (vs moderately experienced)',
    'anesthesiologist_experience_category_no_anesthesiologist': 'No attending anesthesiologist (vs moderately experienced)',
    'resident_experience_category_high': 'Highly experienced resident (vs no resident)',
    'resident_experience_category_low': 'Less experienced resident (vs no resident)',
    'paresthesias_present': 'Paresthesias present during placement (vs none)',
    'number_of_neuraxial_attempts': 'Number of placement attempts (per attempt)',
    'true_procedure_type_incl_dpe_intrathecal': 'Intrathecal catheter (vs conventional epidural)',
    'true_procedure_type_incl_dpe_dpe': 'Dural puncture epidural (vs conventional epidural)',
    'true_procedure_type_incl_dpe_cse': 'Combined spinal-epidural (vs conventional epidural)'
    }

def divide_and_rename_patient_and_procedural_factors(logit_predictor_table):
    patient_factors = [
        "delivery_site_is_bwh",
        "prior_all_catheters_all_enc",
        "prior_failed_catheters_prev_enc",
        "prior_failed_catheters_this_enc",
        "prior_pain_scores_max",
        "composite_psychosocial_problems",
        "CS_hx",                      
        "prior_ob_cmi_scores_max",
        "labor_induction",
        "rom_to_placement_hours",     
        "baby_weight_2196",
        "fetal_position_is_posterior_or_transverse",
        "bmi_end_pregnancy_2044",
        "parity_2048",
        "high_risk_hx",
        "gestational_age_weeks",
        "maternal_age_years",
        "composite_SES_advantage",
        "multiple_gestation",
        "high_risk_current_pregnancy",
        "has_back_problems",
        "has_scoliosis",
        "high_bmi_and_scoliosis",
        "fetal_presentation_is_cephalic",
        "iufd",                     
        "has_dorsalgia",
    ]

    procedural_factors = [
        "true_procedure_type_incl_dpe_intrathecal",
        "true_procedure_type_incl_dpe_cse",
        "resident_experience_category_low",
        "resident_experience_category_high",
        "paresthesias_present",
        "unexpected_delta_lor_squared",  
        "number_of_neuraxial_attempts",
        "anesthesiologist_experience_category_no_anesthesiologist",  
        "anesthesiologist_experience_category_low",
        "true_procedure_type_incl_dpe_dpe",
        "lor_depth",                     
        "anesthesiologist_experience_category_high"
    ]

    # Convert lists to sets for comparison
    patient_set = set(patient_factors)
    procedural_set = set(procedural_factors)
    features_set = set(logit_predictor_table.index) - {'const'}

    # Warn if any feature in your lists is not in the table's index.
    missing_in_table = (patient_set | procedural_set) - features_set
    if missing_in_table:
        warnings.warn("The following features are not in logit_predictor_table: " +
                    ", ".join(missing_in_table))

    # Check that the two groups are mutually exclusive.
    overlap = patient_set & procedural_set
    if overlap:
        raise ValueError("These features appear in both patient_factors and procedural_factors: " +
                        ", ".join(overlap))

    # Check that the groups are complementary, i.e. every feature in the table is in one of the two groups.
    unassigned_features = features_set - (patient_set | procedural_set)
    if unassigned_features:
        raise ValueError("The following features from logit_predictor_table are not assigned to any group: " +
                        ", ".join(unassigned_features))

    patient_factors_filtered = [f for f in patient_factors if f in logit_predictor_table.index]
    procedural_factors_filtered = [f for f in procedural_factors if f in logit_predictor_table.index]

    patient_df = logit_predictor_table.loc[patient_factors_filtered].copy()
    procedural_df = logit_predictor_table.loc[procedural_factors_filtered].copy()

    patient_df = patient_df.rename(index=RENAME_MAP)
    procedural_df = procedural_df.rename(index=RENAME_MAP)

    return patient_df, procedural_df

def print_sklearn_coefficients(feature_names, coefficients):
    # Combine feature names with coefficients into a list of tuples
    coef_pairs = list(zip(feature_names, coefficients))

    # Print the coefficients sorted by absolute magnitude
    print("Coefficients for each feature (sorted by absolute magnitude):")
    for name, coef in sorted(coef_pairs, key=lambda x: abs(x[1]), reverse=True):
        print(f"  {name}: {coef:.4f}")

    # Print the coefficients sorted alphabetically
    print('---------------------------------------------')
    print("Coefficients for each feature (sorted alphabetically):")
    for name, coef in sorted(coef_pairs):
        print(f"  {name}: {coef:.4f}")

def rename_feature_names_onehot_nodrop(feature_names):
    # ML models don't drop the first 'baseline' option in each category
    # So need to correct these
    rename_map_onehot_nodrop = {
    'anesthesiologist_experience_category_high': 'Highly experienced attending anesthesiologist (vs other)',
    'anesthesiologist_experience_category_moderate': 'Moderately experienced attending anesthesiologist (vs other)',
    'anesthesiologist_experience_category_low': 'Minimally experienced attending anesthesiologist (vs other)',
    'anesthesiologist_experience_category_no_anesthesiologist': 'No attending anesthesiologist (vs other)',
    'resident_experience_category_high': 'Highly experienced resident (vs other)',
    'resident_experience_category_low': 'Less experienced resident (vs other)',
    'resident_experience_category_no_resident': 'No resident (vs other)',
    'true_procedure_type_incl_dpe_epidural': 'Conventional epidural catheter (vs other)',
    'true_procedure_type_incl_dpe_intrathecal': 'Intrathecal catheter (vs other)',
    'true_procedure_type_incl_dpe_dpe': 'Dural puncture epidural (vs other)',
    'true_procedure_type_incl_dpe_cse': 'Combined spinal-epidural (vs other)'
    }
    map = RENAME_MAP.copy()
    map.update(rename_map_onehot_nodrop)
    return [map.get(name, name) for name in feature_names]