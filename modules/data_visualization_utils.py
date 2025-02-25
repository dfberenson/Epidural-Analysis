import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

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

        summary_str = f"{median_val:.2f} [{q1:.2f} - {q3:.2f}] (NaN count: {num_table.loc[var_name, 'count_nan']})"
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

def get_contiguous_segments(indices):
    """
    Given a sorted list of indices, returns a list of (start, end) tuples
    for each contiguous block (end is the last index in that block).
    """
    segments = []
    if not indices:
        return segments
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev))
            start = idx
            prev = idx
    segments.append((start, prev))
    return segments

def get_dummies_preserve_order(df, categorical_cols):
    """
    For each column in the DataFrame, if it's in categorical_cols, replace it with its dummy
    columns (using the column name as prefix); otherwise, keep it as is.
    This preserves the original column order.
    """
    result = []
    for col in df.columns:
        if col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            result.append(dummies)
        else:
            result.append(df[[col]])
    return pd.concat(result, axis=1)

def plot_correlation_heatmap_with_related_groups(
    df, 
    drop_columns=None, 
    figsize=(20, 20),
    annot=False, 
    fmt=".2f", 
    cmap="coolwarm",
    title=None, 
    draw_group_lines=False,
    group_line_color='black', 
    group_line_width=2,
    draw_group_boxes=False,
    box_alpha=0.3,         # transparency for the colored boxes (0=transparent, 1=opaque)
    box_horz_multiplier=4.0,   # multiplier for vertical size of extra horizontal axis (bottom boxes)
    box_vert_multiplier=4.0,   # multiplier for horizontal size of extra vertical axis (left boxes)
    additional_groups=None,   # dictionary mapping group name -> list of related column names
    debug=False
):
    """
    This function:
      1. Optionally drops specified columns.
      2. Creates dummy variables for all categorical columns—but replaces each categorical column
         in place so that the original column order is preserved.
      3. Constructs groups:
         - Auto‐generated dummy groups: for each categorical column, all dummy columns 
           (i.e. those with names starting with "<column>_") are grouped together.
         - Additional groups: provided via a dictionary mapping group names to lists of column names.
         These are merged into a combined dictionary.
      4. Computes and plots the correlation matrix using the existing column order.
      5. For every group (over the entire heatmap), it determines the indices of the group’s columns
         in the final order, partitions these indices into contiguous segments, and for each segment draws:
           - Black boundary lines around the block.
           - Extra transparent colored boxes (via extra axes) covering that block.
    
    The column order is not changed from the original DataFrame (with categorical columns replaced by dummies).
    
    If debug is True, intermediate values are printed.
    
    Returns:
      pd.DataFrame: The correlation matrix.
    """
    # --- Data Preparation ---
    df_copy = df.copy()
    if drop_columns:
        df_copy = df_copy.drop(columns=drop_columns, errors='ignore')
    
    # Identify categorical columns.
    categorical_cols = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create dummies while preserving original column order.
    df_transformed = get_dummies_preserve_order(df_copy, categorical_cols)
    final_order = list(df_transformed.columns)
    
    # Build auto-generated dummy groups from categorical columns.
    dummy_groups = {}
    for cat in categorical_cols:
        group = [col for col in final_order if col.startswith(f"{cat}_")]
        if group:
            dummy_groups[cat] = group
            
    # Merge with additional groups.
    all_groups = dummy_groups.copy()
    if additional_groups is not None:
        for grp_name, cols in additional_groups.items():
            if grp_name in all_groups:
                merged = sorted(list(set(all_groups[grp_name] + cols)))
                all_groups[grp_name] = merged
            else:
                all_groups[grp_name] = cols
                
    if debug:
        print("Final column order (preserved):")
        print(final_order)
        print("\nDummy groups:")
        for k, v in dummy_groups.items():
            print(f"{k}: {v}")
        print("\nMerged groups (all_groups):")
        for k, v in all_groups.items():
            print(f"{k}: {v}")
    
    # --- Compute correlation matrix ---
    correlation_matrix = df_transformed.corr()
    
    # --- Plot the main heatmap ---
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        correlation_matrix, 
        annot=annot, 
        fmt=fmt, 
        cmap=cmap, 
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    plt.title(title if title is not None else "Correlation Matrix")
    
    n = len(final_order)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(final_order, rotation=90)
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(final_order, rotation=0)
    
    if debug:
        print(f"\nHeatmap has {n} columns")
    
    # --- Draw black boundaries for each group (over the entire heatmap) ---
    for group_name, group_cols in all_groups.items():
        indices = sorted([final_order.index(col) for col in group_cols if col in final_order])
        if debug:
            print(f"\nGroup '{group_name}' indices: {indices}")
        if not indices:
            continue
        segments = get_contiguous_segments(indices)
        if debug:
            print(f"Group '{group_name}' contiguous segments: {segments}")
        for seg_start, seg_end in segments:
            left = seg_start
            right = seg_end + 1  # boundary extends one cell past the last index
            if draw_group_lines:
                ax.plot([left, right], [left, left], color=group_line_color, lw=group_line_width)
                ax.plot([left, right], [right, right], color=group_line_color, lw=group_line_width)
                ax.plot([left, left], [left, right], color=group_line_color, lw=group_line_width)
                ax.plot([right, right], [left, right], color=group_line_color, lw=group_line_width)
    
    # --- Add extra colored boxes (without text) via extra axes over the entire heatmap ---
    pos = ax.get_position()  # in figure coordinates: [left, bottom, width, height]
    fig = plt.gcf()
    grouped_region_length = n  # entire heatmap spans indices 0 to n
    base_hax_height = 0.06
    base_vax_width = 0.06
    hax_height = base_hax_height * box_horz_multiplier
    vax_width = base_vax_width * box_vert_multiplier
    
    # Extra horizontal axis (for bottom colored boxes).
    hax_left = pos.x0
    hax_width_fig = pos.width
    hax_bottom = pos.y0 - hax_height  # top edge touches bottom of heatmap
    hax = fig.add_axes([hax_left, hax_bottom, hax_width_fig, hax_height])
    hax.set_xlim(0, grouped_region_length)
    hax.set_ylim(0, box_horz_multiplier)
    hax.axis('off')
    
    # Extra vertical axis (for left colored boxes).
    vax_bottom = pos.y0
    vax_height_fig = pos.height
    vax_left = pos.x0 - vax_width  # right edge touches left of heatmap
    vax = fig.add_axes([vax_left, vax_bottom, vax_width, vax_height_fig])
    vax.set_xlim(0, box_vert_multiplier)
    vax.set_ylim(0, grouped_region_length)
    vax.invert_yaxis()
    vax.axis('off')
    
    unique_groups = list(all_groups.keys())
    palette = plt.cm.tab10
    group_colors = {grp: palette(i % 10) for i, grp in enumerate(unique_groups)}
    
    for group_name, group_cols in all_groups.items():
        indices = sorted([final_order.index(col) for col in group_cols if col in final_order])
        if not indices:
            continue
        segments = get_contiguous_segments(indices)
        if debug:
            print(f"Group '{group_name}' segments for boxes: {segments}")
        for seg_start, seg_end in segments:
            rel_start = seg_start  # relative to the entire heatmap (0 to n)
            rel_end = seg_end + 1
            group_width = rel_end - rel_start
            if draw_group_boxes:
                h_rect = Rectangle((rel_start, 0), group_width, box_horz_multiplier,
                                facecolor=group_colors[group_name], edgecolor=None, alpha=box_alpha)
                hax.add_patch(h_rect)
                v_rect = Rectangle((0, rel_start), box_vert_multiplier, group_width,
                                facecolor=group_colors[group_name], edgecolor=None, alpha=box_alpha)
                vax.add_patch(v_rect)
    
    plt.show()
    # return correlation_matrix

def plot_histogram(df, col, bin_space=1, xtick_space=1, xlabel=None, ylabel='Count', title=None):
    """
    Plots a histogram of the specified column in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Column name to plot.
        bins (int or sequence, optional): Number of bins or specific bin edges. Defaults to 10.
    """
    bins = np.arange(df[col].min() // bin_space, df[col].max() + bin_space, bin_space)
    xticks = np.arange(df[col].min() // bin_space, df[col].max() + bin_space, xtick_space)
    plt.figure(figsize=(10, 6))
    plt.hist(df[col].dropna(), bins=bins, edgecolor='black', align='left')
    if xlabel is None:
        plt.xlabel(col)
    else:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    if title is None:
        plt.title(f'Histogram of {col}')
    else:
        plt.title(title)
    plt.show()

def plot_stacked_bar_histogram(df, index_col, value_col, index_col_2 = None, sort_by=None, sort_ascending=False,
                               colormap='Accent', xlabel=None, ylabel='Count', title=None,
                               legend_labels=None, figsize=(6, 6), custom_order=None):
    """
    Creates a stacked bar chart histogram using a crosstab of two specified columns
    from the DataFrame, annotates each bar segment with its percentage of the total,
    and prints the underlying crosstab table.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        index_col (str): Column name to use for the x-axis (rows in the crosstab).
        value_col (str): Column name whose values are counted (columns in the crosstab).
        sort_by (str or int, optional): Column name or position in the crosstab to sort by.
                                        If None and custom_order is not provided, defaults to the first column.
                                        If 'no_sort', no sorting is applied.
        sort_ascending (bool, optional): Whether the sort is ascending. Default is False.
        colormap (str, optional): Colormap to use.
        xlabel (str, optional): Label for the x-axis. Defaults to the name of index_col.
        ylabel (str, optional): Label for the y-axis. Defaults to 'Count'.
        title (str, optional): Title of the plot. Defaults to 'Histogram of <index_col> by <value_col>'.
        legend_labels (list, optional): Labels for the legend. If provided, they will be used
                                        in place of the default legend.
        figsize (tuple, optional): Figure size. Defaults to (6, 6).
        custom_order (list, optional): A list specifying the custom order of categories for the x-axis.
    """
    # Create a crosstab for the two variables
    if index_col_2 is None:
        crosstab = pd.crosstab(df[index_col], df[value_col])
    else:
        crosstab = pd.crosstab([df[index_col], df[index_col_2]], df[value_col])
    
    # Reorder the columns of the crosstab
    crosstab = crosstab[crosstab.sum().sort_values(ascending=False).index]

    if sort_by != 'no_sort':
        # If a custom order is provided, reindex the crosstab accordingly.
        if custom_order is not None:
            crosstab = crosstab.reindex(custom_order, fill_value=0)
        else:
            # If no sort column is provided, sort by the first column
            if sort_by is None:
                sort_by = crosstab.columns[0]
            crosstab = crosstab.sort_values(by=sort_by, ascending=sort_ascending)
    
    # Plot a stacked bar chart
    ax = crosstab.plot(kind='bar', stacked=True, figsize=figsize, colormap=colormap)
    
    # Annotate each bar segment with its percentage of the total bar height
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        # Find the total height of the bar corresponding to this x value
        total_height = sum([patch.get_height() for patch in ax.patches if patch.get_x() == x])
        percentage = (height / total_height * 100) if total_height > 0 else 0
        ax.annotate(f'{percentage:.1f}%', (x + width/2, y + height/2),
                    ha='center', va='center')
    
    # Set plot labels and title
    plt.xlabel(xlabel if xlabel is not None else index_col)
    plt.ylabel(ylabel)
    plt.title(title if title is not None else f'Histogram of {value_col} by {index_col}')
    plt.xticks(rotation=45, ha='right')
    if legend_labels:
        plt.legend(legend_labels)
    plt.tight_layout()
    plt.show()
    
    # Print the crosstab table
    print(f"Table of {index_col} by {value_col}:")
    print(crosstab)

def plot_violin_crosstab_anova(df, index_col, value_col, sort_by=None, sort_ascending=False,
                               colors=None, xlabel=None, ylabel=None, title=None,
                               legend_labels=None, figsize=(10, 6), custom_order=None):
    """
    Creates a violin plot for a continuous variable (value_col) by a categorical variable (index_col),
    prints summary statistics for each group, performs an ANOVA test across the groups, and
    conducts pairwise t-tests between groups.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        index_col (str): Column name for the categorical variable (e.g., resident experience).
        value_col (str): Column name for the continuous variable (e.g., BMI).
        sort_by (str or None, optional): Statistic to sort the groups by (e.g., 'mean', 'std', etc.).
                                         If None and custom_order is not provided, no additional sorting is applied.
        sort_ascending (bool, optional): Whether the sorting is in ascending order. Defaults to False.
        colors (list or str, optional): Color palette for the violin plot.
        xlabel (str, optional): Label for the x-axis. Defaults to index_col.
        ylabel (str, optional): Label for the y-axis. Defaults to value_col.
        title (str, optional): Title of the plot. Defaults to 'Violin Plot of <value_col> by <index_col>'.
        legend_labels (list, optional): Labels for the legend (not typically used for violin plots).
        figsize (tuple, optional): Figure size. Defaults to (10, 6).
        custom_order (list, optional): Custom order for the categories in index_col.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from scipy.stats import f_oneway, ttest_ind
    from itertools import combinations
    
    # Calculate summary statistics for each group (using the continuous variable)
    group_stats = df.groupby(index_col)[value_col].describe()
    
    # Determine the order of groups for plotting
    if custom_order is not None:
        # Reindex the summary statistics to match the custom order
        group_stats = group_stats.reindex(custom_order)
        order = custom_order
    elif sort_by is not None and sort_by in group_stats.columns:
        group_stats = group_stats.sort_values(by=sort_by, ascending=sort_ascending)
        order = group_stats.index.tolist()
    else:
        order = group_stats.index.tolist()
    
    # Print summary statistics for each group
    print("Summary statistics for each group:")
    print(group_stats)
    
    # Prepare data for ANOVA: extract the continuous values for each group in the determined order
    groups = [df[df[index_col] == group][value_col].dropna() for group in order]
    
    # Perform ANOVA if there are at least two groups with data
    if len(groups) > 1 and all(len(g) > 0 for g in groups):
        f_stat, p_value = f_oneway(*groups)
        print(f"\nANOVA results: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
    else:
        print("\nNot enough data in at least two groups for ANOVA.")
    
    # Perform pairwise t-tests for all combinations of groups
    print("\nPairwise t-test results:")
    for group1, group2 in combinations(order, 2):
        data1 = df[df[index_col] == group1][value_col].dropna()
        data2 = df[df[index_col] == group2][value_col].dropna()
        if len(data1) > 0 and len(data2) > 0:
            t_stat, p_val = ttest_ind(data1, data2, equal_var=False)
            print(f"  {group1} vs {group2}: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
        else:
            print(f"  {group1} vs {group2}: Not enough data for t-test.")
    
    # Create the violin plot using the determined order
    plt.figure(figsize=figsize)
    sns.violinplot(x=index_col, y=value_col, data=df, order=order, palette=colors)
    
    # Set plot labels and title
    plt.xlabel(xlabel if xlabel is not None else index_col)
    plt.ylabel(ylabel if ylabel is not None else value_col)
    plt.title(title if title is not None else f'Violin Plot of {value_col} by {index_col}')
    
    plt.tight_layout()
    plt.show()


def plot_binned_errorbar(df, x_axis, y_axis, bin_size, fill_between=True, errorbar=True):
    """
    Plots the average of a y-axis variable (with SEM error bars) 
    versus a binned x-axis variable. The plotted values represent the lower edge of each bin.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_axis (str): The column name to be binned (e.g., 'maternal_age_years').
        y_axis (str): The column name whose mean and SEM are computed (e.g., 'failed_catheter').
        bin_size (numeric): The bin width for grouping the x_axis variable.

    Returns:
        None
    """
    # Drop rows with missing x or y values.
    df_plot = df.dropna(subset=[x_axis, y_axis]).copy()
    
    # Bin the x_axis values.
    # This bins the data by floor-dividing by the bin_size then multiplying back,
    # so each bin label represents the lower edge of that bin.
    bin_column = f'{x_axis}_bin'
    df_plot[bin_column] = (df_plot[x_axis] // bin_size).astype(int) * bin_size
    
    # Group by the binned x values and calculate the mean and standard error of the mean (SEM) for the y_axis.
    grouped = df_plot.groupby(bin_column)[y_axis].agg(['mean', 'sem'])
    
    # Create the plot.
    plt.figure(figsize=(10, 6))
    plt.plot(grouped.index, grouped['mean'], marker='o', linestyle='-', label='Mean')
    if fill_between:
        plt.fill_between(grouped.index,
                        grouped['mean'] - grouped['sem'],
                        grouped['mean'] + grouped['sem'],
                        alpha=0.5, label='SEM')
    if errorbar:
        plt.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'], fmt='o', color='black', capsize=5)
    plt.ylim(bottom=0)
    plt.xlabel(f'{x_axis} (binned by {bin_size})')
    plt.ylabel(f'Average {y_axis}')
    plt.title(f'{y_axis} vs. {x_axis} (binned by {bin_size}) with Error Bars')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_scatter(df, x_axis, y_axis):
    """
    Plots a scatter plot of two specified columns in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        col1 (str): First column name to plot on the x-axis.
        col2 (str): Second column name to plot on the y-axis.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_axis], df[y_axis], alpha=0.5)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'Scatter Plot of {x_axis} vs {y_axis}')
    plt.show()