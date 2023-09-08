import numpy as np
import mlflow
import pandas as pd
import pycaret.classification as clf

def model_score_leads(
    data,
    model_path = 'models/blended_models_final'
    ):
    
    """PyCaret model scoring function.
    
    Args:
        data (DataFrame): Leads data from els.
        db_read_and_process_els_data() function.
        model_path (str, optional): Path to the model. 
        Defaults to 'models/blended_models_final'.

    Returns:
        DataFrame: Leads data with raw prediction scores.
    """
    
    # load the model
    mod = clf.load_model(model_path)
    
    predictions_df = clf.predict_model(mod, data = data, raw_score=True)
    
    return predictions_df 


def mlflow_get_best_run(
    experiment_name,
    n = 1,
    metric = 'metrics.AUC',
    ascending = False,
    tag_source = ["finalize_model", "h2o_automl_model"]
):
    """
    Returns the best run from an MLFlow experiment.

    Args:
        experiment_name (str): MLFlow Experiment Name.
        n (int, optional): The number to return. Defaults to 1.
        metric (str, optional): MLFlow metric to use. Defaults to 'metrics.AUC'.
        ascending (bool, optional): Whether or not to sort the metric ascending or descending.
        Defaults to False.
        tag_source (list, optional): Tag.Source in MLFlow to use in production. Defaults to ["finalize_model", "h2o_automl_model"].

    Returns:
        string: The best MLFlow run id
    """
    
        
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


def mlflow_score_leads(df, run_id):
    """This function scores the leads using an MLflow Run ID to select a model

    Args:
        df (DataFrame): Leads data from els
        run_id (string): An MLFlow Run ID. Recommend to use mlflow_get_best_run()

    Returns:
        DataFrame: A dat aframe with hte leads score column added.
    """

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