import pandas as pd
import numpy as np
import janitor as jn
import plotly.express as px
import pandas_flavor as pf

def cost_calc_monthly_cost_table(
    email_list_size = 1e5,
    email_list_growth_rate = 0.035,
    sales_emails_per_month = 5,
    unsub_rate_per_sales_email = 0.005,
    customer_conversion_rate = 0.05,
    average_customer_value = 2000,
    n_periods = 12
):
    """This function generates a cost table.

    Args:
        email_list_size (_type_, optional): This is the email list size. Defaults to 1e5.
        email_list_growth_rate (float, optional): Monthly email list growth rate. Defaults to 0.035.
        sales_emails_per_month (int, optional): Sales emails per month. Defaults to 5.
        unsub_rate_per_sales_email (float, optional): Unsubscription rate per email. Defaults to 0.005.
        customer_conversion_rate (float, optional): How many customers normally convert from email subscribers. Defaults to 0.05.
        average_customer_value (int, optional): Average customer value. Defaults to 2000.
        n_periods (int, optional): Number of months for our cost table. Defaults to 12.

    Returns:
        _type_: DataFrame: Returns a cost table.
    """
    
    # Period
    period_series = pd.Series(np.arange(0, n_periods), name="period")
    
    cost_table_df = period_series.to_frame()
    
    # Email Size - No Growth

    cost_table_df['email_size_no_growth'] = np.repeat(email_list_size, n_periods)
    
    # Lost Customers - No Growth

    cost_table_df['lost_customers_no_growth'] = cost_table_df['email_size_no_growth'] * unsub_rate_per_sales_email * sales_emails_per_month  
    
    # Cost - No Growth

    cost_table_df['cost_no_growth'] = cost_table_df['lost_customers_no_growth'] * customer_conversion_rate * average_customer_value
    
    # Email Size - With Growth

    cost_table_df['email_size_with_growth'] = cost_table_df['email_size_no_growth'] * ((1+email_list_growth_rate) ** cost_table_df['period'])
    
    # Lost Customers - With Growth

    cost_table_df['lost_customers_with_growth'] = cost_table_df['email_size_with_growth'] * unsub_rate_per_sales_email * sales_emails_per_month
    
    # Cost - With Growth

    cost_table_df['cost_with_growth'] = cost_table_df['lost_customers_with_growth'] * customer_conversion_rate * average_customer_value
        
    
    return cost_table_df

@pf.register_dataframe_method
def cost_total_unsub_cost(cost_table):
    """Generates a summary of the total unsub cost.

    Args:
        cost_table (_type_): Cost table.

    Returns:
        _type_: DataFrame: Returns a summary of the total unsub cost.
    """
    
    summary_df = cost_table[ ['cost_no_growth', 'cost_with_growth'] ] \
        .sum() \
        .to_frame() \
        .transpose()
    
    return summary_df


def cost_simulate_unsub_costs(
    email_list_monthly_growth_rate = [0, 0.035],
    customer_conversion_rate = [0.04, 0.05, 0.06],
    **kwargs
):
    """This function returns a cost simulation to characterize cost uncertainty.

    Args:
        email_list_monthly_growth_rate (list, optional): Monthly email list growth rate to simulate. Defaults to [0, 0.035].
        customer_conversion_rate (list, optional): List of values for customer conversion to simulate. Defaults to [0.04, 0.05, 0.06].

    Returns:
        _type_: DataFrame: Cartesian product of the email list and customer 
        conversion rate is caluclated and total unsubscriber costs are calcualted.
     """
    
    # Parameter Grid
    data_dict = dict(
        email_list_monthly_growth_rate = email_list_monthly_growth_rate,
        customer_conversion_rate = customer_conversion_rate
    )

    parameter_grid_df = jn.expand_grid(others=data_dict)
    
    parameter_grid_df.columns = ['email_list_monthly_growth_rate', 'customer_conversion_rate']
    
    # Temporary Function
    
    # List Comprehension

    def temporary_function(x, y):
        
        cost_table_df = cost_calc_monthly_cost_table(
            email_list_growth_rate=x,
            customer_conversion_rate=y,
            **kwargs
        )
        
        summary_df = cost_total_unsub_cost(cost_table_df)
        
        return summary_df
    
    # List Comprehension
    summary_list = [temporary_function(x, y) for x, y in zip(parameter_grid_df['email_list_monthly_growth_rate'], parameter_grid_df['customer_conversion_rate'])]

    simulation_results_df = pd.concat(summary_list, axis = 0) \
        .reset_index() \
        .drop('index', axis = 1) \
        .merge(parameter_grid_df, left_index=True, right_index=True)
    
    
        
    return simulation_results_df