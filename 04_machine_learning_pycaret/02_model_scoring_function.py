
# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING & APIS
# MODULE 4: MACHINE LEARNING | MODEL LEAD SCORING FUNCTION
# ----

import pandas as pd
import pycaret.classification as clf
import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()

# MODEL LOAD FUNCTION ----

def model_score_leads(
    data,
    model_path = 'models/blended_models_final'
    ):
    
    # load the model
    mod = clf.load_model(model_path)
    
    predictions_df = clf.predict_model(mod, data = data, raw_score=True)
    
    return predictions_df 

model_score_leads(leads_df)

# TEST OUT

import email_lead_scoring as els

leads_df = els.db_read_and_process_els_data()

els.model_score_leads(leads_df, model_path = 'models/xgb_model_tuned')
