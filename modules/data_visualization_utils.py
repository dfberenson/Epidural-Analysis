import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
from sklearn.metrics import roc_curve

FIGURE_FPATH = "C:\\Users\\User\\OneDrive - Mass General Brigham\\Epidural project\\Figures\\"

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

PRESET_GROUPS = {
    'failure': ['failed_catheter','has_subsequent_neuraxial_catheter','has_subsequent_spinal','has_subsequent_airway'],
    'timing': ['placement_to_delivery_hours','rom_thru_delivery_hours','rom_to_placement_hours'],
    'maternal_age_gp': ['maternal_age_years','gravidity_2047','parity_2048'],
    'multiple_gestation_and_labor_induction': ['multiple_gestation','labor_induction'],
    'baby_size': ['gestational_age_weeks','baby_weight_2196'],
    'maternal_size': ['bmi_end_pregnancy_2044', 'bmi_greater_than_40', 'maternal_weight_end_pregnancy_2045', 'bmi_before_pregnancy_2161'],
    'team_composition': ['has_resident','has_anesthesiologist'],
    'team_catheter_counts': ['current_anesthesiologist_catheter_count','current_resident_catheter_count','total_team_catheter_count'],
    'bmi_and_experience': ["high_bmi_and_highly_experienced_resident",    "high_bmi_and_lowly_experienced_resident",    "high_bmi_and_no_resident",    "high_bmi_and_highly_experienced_anesthesiologist"],
    'scoliosis_and_experience': ["scoliosis_and_highly_experienced_resident",    "scoliosis_and_lowly_experienced_resident",    "scoliosis_and_no_resident",    "scoliosis_and_highly_experienced_anesthesiologist"],
    'back_group': ['high_bmi_and_scoliosis','has_scoliosis','has_dorsalgia','has_back_problems'],
    'maternal_risk': ['prior_ob_cmi_scores_max','CS_hx','high_risk_current_pregnancy','high_risk_hx','iufd'],
    'psychosocial_and_ses': ['composite_psychosocial_problems','only_private_insurance','maternal_language_english','marital_status_married_or_partner','country_of_origin_USA','employment_status_fulltime','composite_SES_advantage'],
    'lor': ['lor_depth','predicted_lor_depth','unexpected_delta_lor','unexpected_delta_lor_squared'],
    'pain_and_attempts': ['prior_pain_scores_max','paresthesias_present','number_of_neuraxial_attempts','number_of_spinal_attempts'],
    'prior_catheters': ['prior_failed_catheters_this_enc','prior_failed_catheters_prev_enc','prior_all_catheters_all_enc']
}

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
    
    if additional_groups=='preset':
        additional_groups = PRESET_GROUPS

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
        cbar_kws={'shrink': 0.8},
        center=0
    )

    if title is None:
        title = "Correlation Matrix"

    plt.title(title, fontsize = 20)
    
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
    
    plt.savefig(FIGURE_FPATH + title + '.png', bbox_inches="tight")
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
        title = f'Histogram of {col}'
    plt.title(title)
    plt.savefig(FIGURE_FPATH + title + '.png', bbox_inches = 'tight')
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
    if title is None:
        title = f'Histogram of {value_col} by {index_col}'
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    if legend_labels:
        plt.legend(legend_labels)
    plt.tight_layout()
    plt.savefig(FIGURE_FPATH + title + '.png', bbox_inches = 'tight')
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
    if title is None:
        title = f'Violin Plot of {value_col} by {index_col}'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(FIGURE_FPATH + title + '.png', bbox_inches = 'tight')
    plt.show()


def plot_binned_errorbar(df, x_axis, y_axis, bin_size, fill_between=True, errorbar=True, title=None):
    """
    Plots the average of a y-axis variable (with SEM error bars) 
    versus a binned x-axis variable. The plotted values represent the lower edge of each bin.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        x_axis (str): The column name to be binned (e.g., 'maternal_age_years').
        y_axis (str): The column name whose mean and SEM are computed (e.g., 'failed_catheter').
        bin_size (numeric): The bin width for grouping the x_axis variable.
        title (string): Title for the graph

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
    if title is None:
        title = f'{y_axis} vs. {x_axis} (binned by {bin_size}) with Error Bars'
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(FIGURE_FPATH + title + '.png', bbox_inches = 'tight')
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

def plot_forest_colored_with_markers(
    ax, 
    df, 
    title='Forest Plot', 
    x_label='Odds Ratio (99.9% CI)',
    x_min=0.5, 
    x_max=1.5
    ):
    """
    Plot a forest chart on 'ax' given a DataFrame 'df' with columns:
      - 'OR'
      - 'OR_lower'
      - 'OR_upper'
    
    X-axis is restricted to [x_min, x_max].
    
    Rules:
      - If the center OR is outside [x_min, x_max], skip plotting its dot.
      - If the OR or any part of its CI is beyond [x_min, x_max], place '<' or '>' at that boundary.
      - Print "OR X.XX (L.LL, U.UU)" above each data point in the same color.
      - Color each factor's name on the y-axis to match that factor's color.
    
    Color scheme for significance:
      - red if entire CI > 1
      - blue if entire CI < 1
      - black otherwise
    """

    # Sort by OR if you want smaller/larger ORs in order
    df = df.sort_values('OR')

    # We'll manually set the y-ticks, one row per factor
    y_positions = np.arange(len(df))

    # We won't set the yticklabels yet; we'll do them manually to color each label.
    ax.set_yticks(y_positions)
    # Temporarily set them all to blank
    ax.set_yticklabels([""] * len(df))

    # We'll collect the color for each row, so we can color the labels afterward
    factor_colors = []

    # Plot each factor individually
    for y_pos, (idx, row) in zip(y_positions, df.iterrows()):
        or_val = row['OR']
        ci_low = row['OR_lower']
        ci_high = row['OR_upper']

        # Decide the color based on significance
        if ci_low > 1:
            c = 'red'    # entire CI above 1 => significant risk
        elif ci_high < 1:
            c = 'blue'   # entire CI below 1 => significant protective
        else:
            c = 'black'  # not significant

        factor_colors.append(c)

        # Check if OR or CI extends beyond the plot range
        outside_left = (or_val < x_min) or (ci_low < x_min)
        outside_right = (or_val > x_max) or (ci_high > x_max)

        # If the center OR is out of range, skip the dot
        center_outside = (or_val < x_min) or (or_val > x_max)
        dot_fmt = 'none' if center_outside else 'o'

        # Calculate the full error bar from the center
        left_err = or_val - ci_low
        right_err = ci_high - or_val

        # Plot the error bar (may or may not include the dot)
        ax.errorbar(
            or_val,
            y_pos,
            xerr=[[left_err], [right_err]],
            fmt=dot_fmt,   # skip the dot if center is outside
            color=c,
            ecolor=c,
            capsize=4
        )

        # Place boundary markers if the OR or any part of CI is outside
        if outside_left:
            ax.text(
                x_min, y_pos, '<', 
                va='center', ha='right', color=c, fontsize=14
            )
        if outside_right:
            ax.text(
                x_max, y_pos, '>', 
                va='center', ha='left', color=c, fontsize=14
            )

        # Prepare the label "OR X.XX (L.LL - U.UU)"
        label_str = f"OR {or_val:.2f} ({ci_low:.2f} - {ci_high:.2f})"

        # Place the label just above the data point (or boundary)
        # We'll define a small offset in Y to shift text "above" the marker
        label_offset = 0.2  # Adjust as needed
        label_y = y_pos - label_offset  # axis is inverted => subtract to go "up"
        ax.text(
            1.06, label_y,
            label_str,
            va='bottom',   # text rises from the point
            ha='center',
            color=c,
            fontsize=10
        )

    # Now color each factor name using the same color
    # We already set blank y-ticklabels, so let's manually place them:
    for y_pos, (idx, c) in zip(y_positions, zip(df.index, factor_colors)):
        # We'll place the text a bit left of x_min so it doesn't collide with the plot
        ax.text(
            x_min - 0.05, y_pos,
            idx,
            va='center', ha='right',
            color=c,
            fontsize=20
        )

    # Draw a vertical line at OR=1
    ax.axvline(x=1, color='gray', linestyle='--')

    # Invert y-axis so the top row is at the top
    ax.invert_yaxis()

    # Limit the x-axis
    ax.set_xlim([x_min, x_max])

    # Add labels
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_title(title, fontsize=16)

def show_forest_plots(patient_df, procedural_df):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(25, 12))

    plot_forest_colored_with_markers(
        ax=ax1,
        df=patient_df,
        title='Patient Factors',
        x_label='Odds Ratio (99.9% CI)',
        x_min=0.5,
        x_max=1.5
    )

    plot_forest_colored_with_markers(
        ax=ax2,
        df=procedural_df,
        title='Procedural Factors',
        x_label='Odds Ratio (99.9% CI)',
        x_min=0.5,
        x_max=1.5
    )

    # Create a manual legend for color interpretation:
    protect_marker = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                                label='Significant protective factor')
    ns_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                            label='Not significant')
    risk_marker = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                                label='Significant risk factor')

    fig.legend(
        handles=[protect_marker, ns_marker, risk_marker],
        loc='upper right',
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fontsize=18
    )

    plt.tight_layout()
    plt.savefig(FIGURE_FPATH + 'forest plot' + '.png', bbox_inches='tight')
    plt.show() 

def plot_roc_curve(y_tests, y_probas, test_aucs, labels=None, title="ROC Curves", figsize=(6,6)):
    """
    Plots one or multiple ROC curves on a single figure.
    
    Parameters:
      y_tests (array-like or list): A single array of true labels or a list of arrays for each ROC curve.
      y_probas (array-like or list): A single array of predicted probabilities or a list of arrays for each ROC curve.
      test_aucs (float or list): A single AUC value or a list of AUC values corresponding to each ROC curve.
      labels (str or list, optional): A label or list of labels for each ROC curve. If provided, the AUC will be appended to each label.
      title (str, optional): The title of the plot.
      figsize (tuple, optional): The size of the figure.
    
    Raises:
      ValueError: If the lengths of y_tests, y_probas, and test_aucs do not match, or if labels is provided and its length does not match.
    """
    # Wrap non-list inputs in lists
    if not isinstance(y_tests, (list, tuple)):
        y_tests = [y_tests]
    if not isinstance(y_probas, (list, tuple)):
        y_probas = [y_probas]
    if not isinstance(test_aucs, (list, tuple)):
        test_aucs = [test_aucs]
    if labels is not None and not isinstance(labels, (list, tuple)):
        labels = [labels]
    
    # Error checking for matching lengths
    if not (len(y_tests) == len(y_probas) == len(test_aucs)):
        raise ValueError("y_tests, y_probas, and test_aucs must all have the same number of elements.")
    if labels is not None and len(labels) != len(y_tests):
        raise ValueError("The number of labels must match the number of ROC curves (y_tests).")
    
    plt.figure(figsize=figsize)
    
    # Plot each ROC curve
    for idx, (y_test, y_proba, auc_val) in enumerate(zip(y_tests, y_probas, test_aucs)):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        if labels is not None:
            curve_label = f"{labels[idx]} (AUC = {auc_val:.2f})"
        else:
            curve_label = f"ROC curve (AUC = {auc_val:.2f})"
        plt.plot(fpr, tpr, label=curve_label)
    
    # Plot the reference line for a random classifier
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title(title, fontsize = 20)
    plt.legend(loc="lower right")
    plt.savefig(FIGURE_FPATH + title + '.png', bbox_inches='tight')
    plt.show()


