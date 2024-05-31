import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataframes

df_panda = pd.read_csv('final_data/evaluated_response/evaluation_clinical_panda.csv')
df_gpt3_5 = pd.read_csv('final_data/evaluated_response/evaluation_gpt3.5_turbo.csv')
df_biomistral = pd.read_csv('final_data/evaluated_response/evaluation_biomistral.csv')
df_mistral = pd.read_csv('final_data/evaluated_response/evaluation_mistral.csv')

dfs = [df_panda, df_gpt3_5, df_biomistral, df_mistral]

metrics = ['jaccard_0', 'bert_score_0', 'rouge_1s_0', 'rouge_2s_0',
       'rouge_ls_0']

for metric in metrics:
    panda = []
    gpt3_5_turbo = []
    biomistral = []
    mistral = []
    for i, df in enumerate(dfs):
        # Extract the column corresponding to the current metric
        metric_values = df[metric]
        
        # Append the metric values to the respective lists
        if i == 0:
            panda = metric_values
        elif i == 1:
            gpt3_5_turbo = metric_values
        elif i == 2:
            biomistral = metric_values
        elif i == 3:
            mistral = metric_values
    
    # Create a DataFrame containing the metric values for each model
    metric_df = pd.DataFrame({'Panda': panda, 'GPT3.5 Turbo': gpt3_5_turbo, 'Biomistral': biomistral, 'Mistral': mistral})
    
    # Calculate the correlation matrix
    correlation_matrix = metric_df.corr()
    
    # Print the correlation matrix for the current metric
    print(f'Correlation Matrix for {metric}:')
    # Convert the correlation matrix to LaTeX format
    latex_table = correlation_matrix.to_latex()
    
    # Print the LaTeX table
    print(f'Correlation Matrix for {metric}:')
    print(latex_table)
