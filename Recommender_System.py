import numpy as np
import pandas as pd
import gzip

training_data = pd.read_json(gzip.open('reviews.training.json.gz'))
print(training_data.head())