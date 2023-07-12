# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 1: BUSINESS UNDERSTANDING
# ----


# TEST CALCULATIONS ----

import email_lead_scoring as els

els.cost_calc_monthly_cost_table(
    email_list_size = 1e5, 
    email_list_growth_rate = 0.035,
    sales_emails_per_month = 5,
    unsub_rate_per_sales_email = 0.005,
    customer_conversion_rate = 0.05,
    average_customer_value = 2000,
    n_periods = 12 
)

# ?clf.cost_calc_monthly_cost_table

(els.cost_calc_monthly_cost_table(
    email_list_size = 1e5, 
    email_list_growth_rate = 0.035,
    sales_emails_per_month = 5,
    unsub_rate_per_sales_email = 0.005,
    customer_conversion_rate = 0.05,
    average_customer_value = 2000,
    n_periods = 12
    )
 .cost_total_unsub_cost()
 )

# ?els.cost_total_unsub_cost

els.cost_simulate_unsub_costs()

?els.cost_simulate_unsub_costs

(els.cost_simulate_unsub_costs(
    email_list_monthly_growth_rate=[0.015, 0.025, 0.035]
    )
 .cost_plot_simulated_unsub_costs()
)