import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate confidence interval using bootstrapping
def bootstrap_confidence_interval(data, num_samples, alpha):
    sample_means = []
    for _ in range(num_samples):
        sample = np.random.choice(data, size=len(data), replace=True)
        sample_mean = np.mean(sample)
        sample_means.append(sample_mean)
    lower_percentile = (alpha / 2) * 100
    upper_percentile = 100 - lower_percentile
    lower_bound = np.percentile(sample_means, lower_percentile)
    upper_bound = np.percentile(sample_means, upper_percentile)
    return lower_bound, upper_bound

# Step 1: Read the CSV file
file_mistral = 'final_data/evaluated_response/evaluation_mistral.csv'
file_biomistral = 'final_data/evaluated_response/evaluation_biomistral.csv'
file_gpt3_5turbo = 'final_data/evaluated_response/evaluation_gpt3.5_turbo.csv'
file_panda = 'final_data/evaluated_response/evaluation_clinical_panda.csv'

file_dict = {'mistral': file_mistral, 'biomistral': file_biomistral, 'gpt3.5_turbo': file_gpt3_5turbo,'panda': file_panda}
metrics = ['jaccard', 'bert_score', 'rouge_1s', 'rouge_2s', 'rouge_ls', 'num_mask']
metrics_name_map = {'jaccard': 'Jaccard Score', 'bert_score': 'BERTSCORE', 'rouge_1s': 'ROUGE-1', 'rouge_2s': 'ROUGE-2', 'rouge_ls': 'ROUGE-L', 'num_mask': 'No of Masks'}
# Initialize dictionaries to store average values and confidence intervals
avg_values = {metric: {typ: [] for typ in file_dict.keys()} for metric in metrics}
ci_lower = {metric: {typ: [] for typ in file_dict.keys()} for metric in metrics}
ci_upper = {metric: {typ: [] for typ in file_dict.keys()} for metric in metrics}
masking_levels = [0, 0.25, 0.5, 0.75, 1]

for typ, filename in file_dict.items():
    df = pd.read_csv(filename).drop(columns=['row_id'])
    for metric in metrics:
        for i in range(5):
            avg = df[f'{metric}_{i}'].mean()
            lower, upper = bootstrap_confidence_interval(df[f'{metric}_{i}'], num_samples=100, alpha=0.05)
            avg_values[metric][typ].append(avg)
            ci_lower[metric][typ].append(lower)
            ci_upper[metric][typ].append(upper)

plt.rcParams['font.size'] = 20 

for metric in metrics:
    plt.figure(figsize=(15, 10))
    for typ in file_dict.keys():
        plt.errorbar(masking_levels, avg_values[metric][typ], yerr=[np.array(avg_values[metric][typ])-np.array(ci_lower[metric][typ]), np.array(ci_upper[metric][typ])-np.array(avg_values[metric][typ])], label=typ.capitalize(), marker='o')
    
    plt.xlabel('Masking Level', fontsize=20)
    plt.ylabel(f'Average {metrics_name_map[metric]} ± 95% Confidence Interval', fontsize=20)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=12, ncol=4)
    plt.savefig(f'final_figures/{metrics_name_map[metric]}_variation.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
