import pandas as pd
import numpy as np

# Function: Explore Sales By Category  

def explore_sales_by_category(
    data,
    category = 'country_code',
    sort_by = ['sales', 'prop_in_group']
    ):
    """Returns a dataframe with sales aggregated in terms of sales and average by specified category.

    Args:
        data (DataFrame): _description_
        category (str, optional): Categorical variable to group sales by. Defaults to 'country_code'.
        sort_by (list, optional): Variables to sort output DataFrame by. Defaults to ['sales', 'prop_in_group'].

    Returns:
        _type_: Pandas DataFrame with sales aggregated in terms of sales and average by specified category.
    """
    
    # Handle sort by
    if (type(sort_by) is list):
        sort_by = sort_by[0]
    
    # Handle data manipulation
    ret = (data
    .groupby(category)
    .agg(
        dict(
            made_purchase = ['sum', 'mean']
            )
        )
    .set_axis(['sales', 'prop_in_group'], axis='columns')
    .assign(
        prop_overall=lambda x: x['sales'] / sum(x['sales'])
    )
    .sort_values(by=sort_by, ascending=False)
    .assign(
        prop_cum_sum = lambda x:
                x['prop_overall'].cumsum()
     )
    )
    
    return ret


# Function: Explore Sales by Numeric Feature

def explore_sales_by_numeric(
    data,
    numeric_feature = ['tag_count'],
    q = [0.10, 0.50, 0.90]
    ):
    """Return DataFrame with sales quantiled by specified numeric feature.

    Args:
        data (DataFrame): DataFrame to perform operation on.
        numeric_feature (list, optional): Numeric feature to get quantile bins from. Defaults to ['tag_count'].
        q (list, optional): Quantile bins. Defaults to [0.10, 0.50, 0.90].

    Returns:
        Pandas DataFrame: Sales quantiled by specified numeric feature.
    """
    
    if (type(numeric_feature) is list):
        feature_list = ['made_purchase', *numeric_feature]
    else:
        feature_list = ['made_purchase', numeric_feature]
    
    
    ret = (data
     [feature_list]
    .groupby('made_purchase')
    .quantile([0.10, 0.50, 0.90])
    )
    
    return ret