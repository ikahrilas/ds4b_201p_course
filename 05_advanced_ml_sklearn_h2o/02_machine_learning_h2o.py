# BUSINESS SCIENCE UNIVERSITY
# COURSE: DS4B 201-P PYTHON MACHINE LEARNING
# MODULE 5: ADVANCED MACHINE LEARNING 
# PART 2: H2O AUTOML
# ----

import pandas as pd
import numpy as np

import h2o
from h2o.automl import H2OAutoML

import email_lead_scoring as els

# Collect Data

leads_df = els.db_read_and_process_els_data()


# 1.0 H2O PREPARATION

# Initialize H2O



# Convert to H2O Frame




# Prep for AutoML




# 2.0 RUN H2O AUTOML ----

# H2OAutoML




# Save / load the model




# CONCLUSIONS ----
# 1. H2O AutoML handles A LOT of stuff for you (preprocessing)
# 2. H2O is highly scalable
# 3. (CON) H2O depends on Java, which adds another complexity when you take your model into production
