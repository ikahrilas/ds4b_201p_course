# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 2: DATA UNDERSTANDING
# PART 2: DATA IMPORT FUNCTIONS
# ----

# LIBRARIES ----

import pandas as pd
import numpy as np
import sqlalchemy as sql


# IMPORT RAW DATA ----

# Read & Combine Raw Data

def db_read_els_data(
    conn_string='sqlite:///00_database/crm_database.sqlite'
    ):
    
    # Connect to engine
    
    engine = sql.create_engine(conn_string)
    conn = engine.connect()
    
    # Raw data collect
    
    with engine.connect() as conn:
        
        # subscribers
        
        subscribers_df = pd.read_sql(
            sql="""
            SELECT * FROM Subscribers
            """,
            con=conn
        )
        
        subscribers_df = subscribers_df.astype({
            'mailchimp_id': 'int',
            'member_rating': 'int',
            'optin_time': 'datetime64',
        })
        
        # tags
        
        tags_df = pd.read_sql(
            sql=f"""
            select * from Tags
            """,
            con=conn
        )
        
        tags_df = tags_df.astype({
            'mailchimp_id': 'int'
        })
        
        # transactions
        
        transactions_df = pd.read_sql(
            sql=f"""
            select * from Transactions
            """,
            con=conn
        )
        
        transactions_df = transactions_df.astype({
            'purchased_at': 'datetime64',
            'product_id': 'int'
        })
        
    # Merge tag counts
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
    
    # Merge target variable
    
    emails_made_purchase = transactions_df['user_email'].unique()
    
    subscribers_joined_df = (subscribers_joined_df
    .assign(
        made_purchase = 
            subscribers_joined_df.user_email
            .isin(emails_made_purchase)
            .astype('int')
     )
    )
    
    
    return subscribers_joined_df

db_read_els_data()

# Read Table Names



# Get Raw Table





# TEST IT OUT -----



