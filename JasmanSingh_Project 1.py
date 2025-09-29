# Project 1
# Jasman Singh, 501180039
# Due October 6, 2025

# Step 1
import pandas as pd
data = pd.read_csv("Project 1 Data.csv")

# Step 2
import matplotlib.pyplot as plt
import numpy as np
#print(data.head())
#print(data.info())
#print(data.describe())
classes = data['Step'].unique()
print("Classes in dataset:", classes)