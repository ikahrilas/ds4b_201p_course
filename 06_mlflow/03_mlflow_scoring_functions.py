# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 6: MLFLOW 
# PART 3: PREDICTION FUNCTION 
# ----

import pandas as pd
import mlflow
import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()


# 1.0 GETTING THE BEST RUN FOR PRODUCTION ----

EXPERIMENT_NAME = 'automl_lead_scoring_1'

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

experiment_id = experiment.experiment_id

logs_df = mlflow.search_runs(experiment_id)

logs_df \
    .query('`tags.Source` in ["finalize_model", "h2o_automl_model"]') \
    .sort_values('metrics.AUC', ascending = False) \
    ['run_id'] \
    .values[0]

# Function

def mlflow_get_best_run(
    experiment_name,
    n = 1,
    metric = 'metrics.AUC',
    ascending = False,
    tag_source = ["finalize_model", "h2o_automl_model"]
):
        
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    logs_df = mlflow.search_runs(experiment_id)
    
    # handle different naming conventions
    if experiment_name == 'automl_lead_scoring_1':
        logs_df = (logs_df
         .rename(columns = 
                 {'tags.source': 'tags.Source',
                  'metrics.auc': 'metrics.AUC'})
         )

    best_run_id = (logs_df
    .query(f'`tags.Source` in {tag_source}')
    .sort_values(metric, ascending = False)
    ['run_id']
    .values[n-1]
    )
    
    return best_run_id

mlflow_get_best_run('automl_lead_scoring_1')

mlflow_get_best_run(
    'automl_lead_scoring_1',
    n = 3,
    metric = 'metrics.AUC'
)



# 2.0 PREDICT WITH THE MODEL (LEAD SCORING FUNCTION)


# Load model as a PyFuncModel.

# H2O

# Sklearn / Pycaret (Extract)

# Function



# 3.0 TEST WORKFLOW ----


