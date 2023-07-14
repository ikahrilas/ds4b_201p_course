# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA UNDERSTANDING
# PART 1: DATA UNDERSTANDING & KPIS
# ----

# GOAL: ----
# - Saw high costs, feedback showed problems
# - Now need to work with departments to collect data and develop project KPIs

# LIBRARIES ----

# Data Analysis:
import pandas as pd
import numpy as np
import plotly.express as px

# New Libraries:
import sweetviz as sv
import sqlalchemy as sql

# Email Lead Scoring: 
import email_lead_scoring as els


# ?els.cost_calc_monthly_unsub_cost_table

els.cost_simulate_unsub_costs(
    email_list_monthly_growth_rate=np.linspace(0, 0.03, 5),
    customer_conversion_rate=np.linspace(0.4, 0.6, 3),
    sales_emails_per_month=5,
    unsub_rate_per_sales_email=0.001,
    email_list_size=1e5
) \
    .pipe(func=els.cost_plot_simulated_unsub_costs)


# 1.0 CONNECTING TO SQLITE DATABASE ----

engine = sql.create_engine('sqlite:///00_database/crm_database.sqlite')

conn = engine.connect()

table_names = sql.inspect(engine).get_table_names()

table_names

# 2.0 COLLECT DATA ----

# Products ----

products_df = pd.read_sql(
    sql="""
    SELECT * FROM products
    """,
    con=conn
)

products_df.head()

products_df.shape

products_df.info()

## change product_id to int
products_df = (products_df
 .assign(
     product_id=lambda x: x['product_id'].astype('int'),
 )
)

# Subscribers ----
table_names[1]

subscribers_df = pd.read_sql(
    sql="""
    SELECT * FROM Subscribers
    """,
    con=conn
)

subscribers_df.head()

subscribers_df.shape

subscribers_df.info()

memory_usage = subscribers_df.memory_usage(deep=True).sum()

# change mailchimp_id and member_rating to int
subscribers_df = (subscribers_df
 .assign(
     mailchimp_id = lambda x: x.mailchimp_id.astype('int'),
     member_rating = lambda x: x.member_rating.astype('int'),
     optin_time = lambda x: x.optin_time.astype('datetime64')
 )
)

subscribers_df.info()

memory_usage_post = subscribers_df.memory_usage(deep=True).sum()

# quite a bit of memory saved by switching data types
memory_usage - memory_usage_post

# Tags ----
table_names[2]

tags_df = pd.read_sql(
    sql=f"""
    select * from {table_names[2]}
    """,
    con=conn
)

tags_df.info()

tags_df['mailchimp_id'] = tags_df['mailchimp_id'].astype('int')

# Transactions ----
transactions_df = pd.read_sql(
    sql=f"""
    select * from {table_names[3]}
    """,
    con=conn
)

transactions_df.head()

transactions_df.shape

transactions_df.info()

transactions_df = (transactions_df
 .assign(
     purchased_at = lambda x: x.purchased_at.astype('datetime64'),
     product_id = lambda x: x.product_id.astype('int'),
 )
)

# Website ----

website_df = pd.read_sql(
    sql=f"""
    select * from {table_names[4]}
    """,
    con=conn
)

website_df.head()

website_df.shape

website_df.info()

# here another way to change data types on multiple columns
## could also use for and define the type conversion function
website_df = website_df.astype({
    'date': 'datetime64',
    'pageviews': 'int',
    'organicsearches': 'int',
    'sessions': 'int',
})

website_df.info()

# Close Connection ----

conn.close()

# - Note: a better practice is to use `with`
with engine.connect() as conn:
    website_df = pd.read_sql("select * from Website", conn)
    
    website_df = website_df.astype({
        'date': 'datetime64',
        'pageviews': 'int',
        'organicsearches': 'int',
        'sessions': 'int',
    })


# 3.0 COMBINE & ORGANIZE DATA ----
# - Problem is related to probability of purchase from email list
# - Need to understand what increases probability of purchase
# - Learning Labs could be a key event
# - Website data would be interesting but can't link it to email
# - Products really aren't important to our initial question - just want to know if they made a purchase or not and identify which are likely

# Make Target Feature

subscribers_df

emails_made_purchase = transactions_df['user_email'].unique()

# using assign

subscribers_df = (subscribers_df
 .assign(
     made_purchase = 
        subscribers_df.user_email
        .isin(emails_made_purchase)
        .astype('int')
 )
)

# subscribers_df['made_purchase'] (subscribers_df['user_email']
#  .isin(emails_made_purchase)
#  .astype('int')
# )

# Who is purchasing?

count_made_purchase = subscribers_df['made_purchase'].sum()
total_subscribers = len(subscribers_df['made_purchase'])

count_made_purchase / total_subscribers

# By Geographic Regions (Countries)

# (subscribers_df
#  .groupby('country_code')
#  .agg(
#      dict(
#          made_purchase = ['sum', lambda x: x.sum() / len(x)]
#          )
#     )
#  .set_axis(['sales', 'prop_in_group'], axis='columns')
#  .assign(
#      prop_overall=lambda x: x['sales'] / sum(x['sales'])
#  )
# )

# using mean in agg function instead of lambda

by_geography_df = (subscribers_df
 .groupby('country_code')
 .agg(
     dict(
         made_purchase = ['sum', 'mean']
         )
    )
 .set_axis(['sales', 'prop_in_group'], axis='columns')
 .assign(
     prop_overall=lambda x: x['sales'] / sum(x['sales'])
 )
 .sort_values(by='sales', ascending=False)
 .assign(
     prop_cum_sum = lambda x:
            x['prop_overall'].cumsum()
 )
)

# - Top 80% countries 

(by_geography_df
 .query('prop_cum_sum <= 0.8')
)


# - High Conversion Countries (>8% conversion)

(by_geography_df
 .query('prop_in_group >= 0.08')
)

by_geography_df.quantile([0.10, 0.50, 0.90])

by_geography_df.mean()

# By Tags (Events)

tags_df.sample(5)

(tags_df
 .groupby('tag')
 .agg(
     dict(tag = 'count')
     )
)

user_events_df = (tags_df
 .groupby('mailchimp_id')
 .agg({
     'tag':'count'
 })
 .set_axis(['tag_count'], axis='columns')
 .reset_index()
)

subscribers_joined_df = (subscribers_df
 .merge(user_events_df, 
        on='mailchimp_id', 
        how='left')
 .fillna({
    'tag_count': 0
 })
 .astype({
     'tag_count': 'int'
 })
)

# Analyzing tag count proportions

(subscribers_joined_df
 .groupby('made_purchase')
 [['member_rating', 'tag_count']]
 .quantile([0.10, 0.50, 0.90])
)

# 4.0 SWEETVIZ EDA REPORT

report = sv.analyze(
    subscribers_joined_df,
    target_feat='made_purchase'
)

report.show_html(
    filepath='02_data_understanding/subscriber_eda_report.html',
)

# 5.0 DEVELOP KPI'S ----
# - Reduce unnecessary sales emails by 30% while maintaing 99% of sales
# - Segment list - apply sales (hot leads), nuture (cold leads)

# EVALUATE PERFORMANCE -----

(subscribers_joined_df
 [['made_purchase', 'tag_count']]
 .groupby('made_purchase')
 .agg(
     mean_tag_count = ('tag_count', 'mean'),
     median_tag_count = ('tag_count', 'median'),
     count_subsribers = ('tag_count', 'count')
 )
)

# WHAT COULD BE MISSED?
# - More information on which tags are most important






