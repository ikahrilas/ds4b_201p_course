# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA UNDERSTANDING: EXPLORATORY
# 
# ----

# LIBRARIES ----

import pandas as pd
import numpy as np

# Data

import email_lead_scoring as els

df = els.db_read_els_data()

# Function: Explore Sales By Category  

def explore_sales_by_category(
    data,
    category = 'country_code',
    sort_by = ['sales', 'prop_in_group']
    ):
    
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


explore_sales_by_category(df, 'member_rating', 'prop_in_group')    

# Function: Explore Sales by Numeric Feature

def explore_sales_by_numeric(
    data,
    numeric_feature = ['tag_count'],
    q = [0.10, 0.50, 0.90]
    ):
    
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
    
explore_sales_by_numeric(df, 
                         ['tag_count', 'member_rating'], 
                         [0.05, 0.25, 0.50, 0.75, 0.95])
    

#  TEST THEM OUT ---- 

import email_lead_scoring as els

df = els.db_read_els_data()

els.explore_sales_by_category(df, 'member_rating', 'prop_in_group')

els.explore_sales_by_numeric(df)