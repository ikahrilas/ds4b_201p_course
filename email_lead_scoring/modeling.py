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