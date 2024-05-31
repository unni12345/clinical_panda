import pandas as pd
import json
import csv
import time
import numpy as np
import torch
import re
import shutil
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import argparse
import subprocess

class DiagnosticReasoner(object):
    def __init__(self, model_repo_id="models/biomistral", adapter_repo=None, prompt_template="<s>[INST] {} [/INST]", num_beams=4,
                 input_filename="test.csv",
                 gcloud_folder="clinical_panda/"):
        self.model_repo_id = model_repo_id
        self.adapter_repo = adapter_repo
        self.num_beams = num_beams
        self.input_filename = input_filename
        self.gcloud_folder = gcloud_folder
        self.prompt_template = prompt_template
        self.df_input = self.setup()
        self.setup_model()

    def setup(self):
        # drive.mount('/content/drive')
        filename = self.gcloud_folder + self.input_filename
        df_input = pd.read_csv(filename)
        return df_input

    def setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_repo_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_repo_id)
        if self.adapter_repo != None:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_repo)
        self.model = self.model.to('cuda')


    def get_response(self, text):
        # encoded_text = tokenizer(text, return_tensors="pt")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        text = f"<s>[INST] {text.strip()} [/INST]"
        # text = self.prompt_template.format(text)

        encoded_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, return_attention_mask=True)

        encoded_text = encoded_text.to('cuda')
        # Use beam search decoding with output scores
        outputs = self.model.generate(
            **encoded_text,
            max_new_tokens=256,
            temperature=0,
            num_beams=self.num_beams,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True
        )

        # Extract sequences, scores, and beam indices
        sequences = outputs.sequences
        scores = outputs.scores
        beam_indices = outputs.beam_indices

        # Get the generated tokens from each sequence, removing the input_ids
        sequences = sequences[:, encoded_text['input_ids'].shape[1]:][:]

        # Calculate transition scores for each beam
        transition_scores = self.model.compute_transition_scores(
            sequences, scores, beam_indices, normalize_logits=True
        )

        # Get individual sentence probabilities
        sentence_probs = []

        for i in range(len(sequences)):
            # Decode tokens, skipping special ones for accurate punctuation
            raw_sentence = self.tokenizer.decode(sequences[i], skip_special_tokens=True)

        return {'response': raw_sentence}
    
    def clean_gpu(self):
        try:
            subprocess.run(['nvidia-smi', '--gpu-reset'], check=True)
            print("GPU reset successfully.")
        except subprocess.CalledProcessError as e:
            print("Error resetting GPU:", e)

    def evaluate(self):
        # getting response one by one
        t1 = time.time()
        inference_count = 0
        data_prompt = {'row_id': [], 'gpt4_turbo_response': [], 'explanation_0': []
                       , 'explanation_1': [], 'explanation_2': [], 'explanation_3': [], 'explanation_4': [],
                       'prompt_0': [], 'prompt_1': [], 'prompt_2': [], 'prompt_3': [], 'prompt_4': [] }

        for idx, row in self.df_input.iterrows():
            data_prompt['row_id'].append(row['row_id'])
            data_prompt['gpt4_turbo_response'].append(row['gpt4_turbo_response'])

            for i in range(5):
                prompt = row[f'prompt_{i}']
                response = self.get_response(prompt)
                inference_count += self.num_beams
                data_prompt[f'explanation_{i}'].append(response['response'])
                data_prompt[f'prompt_{i}'].append(prompt)

                print(f"row = {idx}, masking_level={i} response: ")
                print("prompt: ", prompt)
                print(response["response"])
                print("########################################################################")


        t2 = time.time()
        print("data_prompt: ")
        print(data_prompt)

        df_prompt = pd.DataFrame(data=data_prompt)
        tot = time.time()
        file_name = f'clinical_panda_results/masked/final/masked_response_trial_{self.input_filename[:-4]}_{str(tot)}.csv'
        df_prompt.to_csv(file_name)
        # shutil.copy(file_name, self.gcloud_folder)

        file_name = f"new_data/masked_trial_{self.input_filename[:-4]}_{str(tot)}.txt"
        with open(file_name, "w") as f:
        # Concatenate the strings and write them to the file
            f.write("Time spent: {} seconds\n".format(t2 - t1))
            f.write("Number of inferences: {}\n".format(inference_count))
        # shutil.copy(file_name, self.gcloud_folder)

        print("time spend: ",t2-t1)
        print("No of inferences: ", inference_count)
        return True
    
####################################################################################################
# CUDA_LAUNCH_BLOCKING=1 python3 clinical_panda_script/diagnostic_reasoner_masked.py --model_repo "mistralai/Mistral-7B-Instruct-v0.1" --adapter_repo "None" --filename "masked_texts_mistral.csv"
# CUDA_LAUNCH_BLOCKING=1 python3 clinical_panda_script/diagnostic_reasoner_masked.py --model_repo "BioMistral/BioMistral-7B" --adapter_repo "None" --filename "masked_texts_biomistral.csv"
# CUDA_LAUNCH_BLOCKING=1 python3 clinical_panda_script/diagnostic_reasoner_masked.py --model_repo "BioMistral/BioMistral-7B" --adapter_repo "Yes" --filename "masked_texts_clinical_panda.csv"
    
def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--adapter_repo', type=str, required=True, help='Adapter repository')
    parser.add_argument('--model_repo', type=str, required=True, help='Model repository')
    parser.add_argument('--filename', type=str, required=True, help='File Name of input masked csv file')

    args = parser.parse_args()
    adapter_repo = None if args.adapter_repo == "None" else "biomistral_instruct_explanation_full"
    model_repo = args.model_repo
    input_filename = args.filename

    # model_repo_id="models/biomistral"
    prompt_template = "<s>[INST] {} [/INST]"
    num_beams = 4
    gcloud_folder = "final_data/masked/"
    
    dr = DiagnosticReasoner(model_repo, adapter_repo, prompt_template, num_beams, input_filename, gcloud_folder)
    dr.evaluate()
    
    # Call the clean_gpu function to reset the GPU
    dr.clean_gpu()

if __name__ == "__main__":
    main()

