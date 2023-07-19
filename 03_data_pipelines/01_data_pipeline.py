# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 3: DATA PROCESSING PIPELINE 
# ----

# LIBRARIES ----

# Core
import pandas as pd
import numpy as np
# EDA 
import re
import janitor as jn
import sweetviz as sv
# Email Lead Scoring
import email_lead_scoring as els

# Recap  ---
# - We see some promise in the data
# - Now we need to continue to develop features
# - Prepare for Machine Learning

leads_df = els.db_read_els_data()

leads_df.info()

leads_df.sample(5).transpose()

# 1.0 FEATURE EXPLORATION ----

# High Cardinalaity
## object type columns high number of unique values

(els.explore_sales_by_category(
    leads_df,
    "country_code",
    "prop_in_group"
    )
 .sort_values('sales', ascending = False)
 .query('sales > 5')
)

# Ordinal Feature

els.explore_sales_by_category(
    data=leads_df,
    category='member_rating',
    sort_by='sales'
)

# Interaction

els.explore_sales_by_numeric(
    data=leads_df,
    numeric_feature=['tag_count', 'member_rating']
)

# 2.0 BUILDING ENGINEERED FEATURES ----

# Date Features

date_max = leads_df['optin_time'].max()
date_min = leads_df['optin_time'].min()

time_range = date_min - date_max

time_range.days

(leads_df['optin_time'] - date_max).dt.days

leads_df = (leads_df
 .assign(optin_days = 
         (leads_df['optin_time'] - date_max).dt.days
 )
)

# Email Features

leads_df = (leads_df
 .assign(
     email_provider = 
     leads_df.user_email.str.split('@', expand = True)[1]
 )
)

# Activity Features (rate features)

leads_df = (leads_df
 .assign(
     tag_count_by_optin_day = leads_df['tag_count'] / abs(leads_df['optin_days'] - 1)
 )
)

# 3.0 ONE-TO-MANY FEATURES  ----
# - TAGS: 1 Customer can have many tags

# Specific Tag Features (Actions)

tags_df = els.db_read_raw_els_table('Tags')

tags_wide_leads_df = (tags_df
 .assign(
     value = 1,
     mailchimp_id = tags_df.mailchimp_id.astype('int')
 )
 .pivot(
     index='mailchimp_id',
     columns='tag',
     values='value'
 )
 .fillna(0)
 .clean_names()
 .change_type(lambda df_: df_.select_dtypes('float64').columns, 'int')
)

# Merge Tags

tags_wide_leads_df.columns = (tags_wide_leads_df
 .columns
 .to_series()
 .apply(lambda x: f"tag_{x}")
 .to_list()
)

tags_wide_leads_df = tags_wide_leads_df.reset_index()

leads_tags_df = (leads_df
 .merge(
     tags_wide_leads_df,
     how = 'left'
     )
)

# Fill NA selectively

def fill_na_regex(data, regex, value = 0, **kwargs):
    for col in data.columns:
        if re.match(pattern=regex, string=col):
            data[col] = data[col].fillna(value = value, **kwargs)
    return data
    
leads_tags_df = fill_na_regex(data = leads_tags_df, regex = 'tag_')

# 4.0 ADJUSTING FEATURES ----
# - Country Code: Has high cardinality

# High Cardinality Features: Country Code

countries_to_keep = (els.explore_sales_by_category(
leads_tags_df,
category='country_code'
)
.query('sales >= 6')
.index
.to_list()
)

leads_tags_df['country_code'] = (leads_tags_df
 ['country_code']
 .apply(lambda x: x if x in countries_to_keep else 'other')
)

# more performant option using np.where and assign
(leads_tags_df
 .assign(
     country_code = 
     np.where(leads_tags_df.country_code.isin(countries_to_keep), leads_tags_df.country_code, 'other')
 )
)

# 5.0 EXPLORATORY REPORT PART 2

# Examine with Sweetviz

report = sv.analyze(
    source=leads_df,
    target_feat='made_purchase',
    feat_cfg = sv.FeatureConfig(
        skip = ['mailchimp_id', 'user_full_name', 'user_email', 'optin_time'],
        force_cat = ['member_rating', 'country_code', 'email_provider']
    )
)

report.show_html(
    filepath='03_data_pipelines/subscriber_report_pipeline.html'
)


# CONCLUSIONS ----

# - Member Rating: 5 has 14% of purchasers within group
# - Country Code: US getting 11% conversion, AU getting 12%
# - Tag Count: 
#    - At zero tags, almost no chance of purchase
#    - At 5 tags, about 10% chance of purchase
#    - At 10 tags, about 35% chance of purchase
# - Optin Days: After 100 days, increases to 2% to 4-5% likelihood
# - Ratio of Tag Counts to Length of Time: Goes from zero to 15% at 0.10 (1 tag for every 10 days on list)
# - Attending Learning Labs & Webinars: Increases from 2% to 10%



