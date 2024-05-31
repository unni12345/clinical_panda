## Dependencies:
# Create a new directory for your project
mkdir my_project
cd my_project

# Create a virtual environment
python3 -m venv my_env

# Activate the virtual environment
source my_env/bin/activate  # For Linux/Mac
my_env\Scripts\activate.bat  # For Windows

# Install required packages
pip install pandas numpy scikit-learn matplotlib openai torch transformers peft

## Getting Final Results from Processed Files:
python3 final_scripts/correlation_of_models.py
python3 final_scripts/plot_final_with_confidence.py


## Training can be done using ipynb file on colab:
final_scripts/train_clinical_panda.ipynb

## Getting data and permissions 
# DDXPLUS dataset from : https://github.com/mila-iqia/ddxplus/tree/main
- unzip release_test_patients.zip
- move release_test_patients.csv to final_data folder

# Getting access to medcat model: https://github.com/CogStack/MedCAT -> Need to request access to the model
- place the model like:
- models/umls_sm_pt2ch_533bab5115c6c2d6.zip

## Adapter of Clinical Panda after training: should be placed in the following directory -> biomistral_instruct_explanation_full

You can access the zip file from: https://zenodo.org/records/11071023?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImEwNzAyODQ5LTUxODMtNDMyMS1hMWU1LTY2NTdhMTdiZWVjNSIsImRhdGEiOnt9LCJyYW5kb20iOiJiMzNiNzBmNDFmODI4MTU3MzhmMmI5YzMzYjU0OWFlNCJ9.KcK-0L4dv0rSXhNHRSKtEkwXzObhSHDudlivRLXHfbJ33JjovVPUUlUCigGiYdSBlYNhw5xewsIlAKihRam7KQ

### Code to tryout Clinical Panda
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B")
adapter_repo = "biomistral_instruct_explanation_full"
peft_model = PeftModel.from_pretrained(model, adapter_repo)
peft_model = peft_model.to('cuda')

# Prepare input text
input_text = "Admission Note: Patient Details: 
Age: 21
Sex: Female

Chief Complaint (CC): 
The patient presents with a persistent cough.

History of Present Illness (HPI): 
The patient has been experiencing a persistent cough, shortness of breath, and a wheezing sound when exhaling. The symptoms have been ongoing for an unspecified duration. The patient lives in a big city and has not traveled out of the country in the last 4 weeks. The patient has a family history of asthma and has been previously diagnosed with chronic sinusitis. The patient has also used a bronchodilator in the past.

Past Medical History (PMH): 
The patient has a history of chronic sinusitis and asthma.

Medications and Allergies: 
The patient has not provided any information on current medications or known allergies.

Physical Examination (PE): 
Vitals: Not provided.
General: The patient appears to be in distress due to difficulty in breathing.
The patient's breathing sounds are abnormal with audible wheezing."

# Tokenize input text
input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')

# Generate text
output_ids = peft_model.generate(input_ids, max_length=256)

# Decode generated text
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated text:", output_text)


## OPENAI starter: https://platform.openai.com/docs/quickstart

## To run diagnostic_reasoner_masked.py -> This generates the Explanation with various level of masking for mistral, biomistral and clinical panda
python3 clinical_panda_script/diagnostic_reasoner_masked.py --model_repo "mistralai/Mistral-7B-Instruct-v0.1" --adapter_repo "None" --filename "masked_texts_mistral.csv"
python3 clinical_panda_script/diagnostic_reasoner_masked.py --model_repo "BioMistral/BioMistral-7B" --adapter_repo "None" --filename "masked_texts_biomistral.csv"
python3 clinical_panda_script/diagnostic_reasoner_masked.py --model_repo "BioMistral/BioMistral-7B" --adapter_repo "Yes" --filename "masked_texts_clinical_panda.csv"
 
