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

run_id = mlflow_get_best_run('automl_lead_scoring_1')

logged_model = f'runs:/{run_id}/model'

loaded_model = mlflow.pyfunc.load_model(logged_model)

loaded_model.predict(leads_df)['p1']

# Sklearn / Pycaret (Extract)

run_id = mlflow_get_best_run('email_lead_scoring_0')

logged_model = f'runs:/{run_id}/model'

loaded_model = mlflow.pyfunc.load_model(logged_model)

df = (leads_df
 .drop(columns=[
     'mailchimp_id', 
     'user_full_name', 
     'user_email', 
     'optin_time', 
     'email_provider',
     'made_purchase'
     ],
       axis = 'columns')
 .assign(
     member_rating = leads_df.member_rating.astype(str)
 )
)

loaded_model._model_impl.predict_proba(df)[:,1]

# Function

def mlflow_score_leads(df, run_id):

    logged_model = f'runs:/{run_id}/model'
    
    print(f'The logged model is: {logged_model}')
    
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    # Predict
    
    try:
        predictions_array = loaded_model.predict(pd.DataFrame(df))['p1']
    except:
        df1 = (df
        .drop(columns=[
            'mailchimp_id', 
            'user_full_name', 
            'user_email', 
            'optin_time', 
            'email_provider',
            'made_purchase'
            ],
            axis = 'columns')
        .assign(
            member_rating = leads_df.member_rating.astype(str)
         )
        )
        
        predictions_array = loaded_model._model_impl.predict_proba(df1)[:,1]
        
    predictions_series = pd.Series(predictions_array, name = 'Score')
    
    ret = pd.concat([predictions_series, df], axis = 'columns')
    
    return ret

mlflow_score_leads(
    leads_df, 
    mlflow_get_best_run('email_lead_scoring_0')
)


# 3.0 TEST WORKFLOW ----

import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()

best_run_id = els.mlflow_get_best_run('email_lead_scoring_0')

els.mlflow_score_leads(df = leads_df, run_id = best_run_id)
