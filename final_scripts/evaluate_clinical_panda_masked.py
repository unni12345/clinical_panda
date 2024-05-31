from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd
import json
import csv
import time
import shutil
import textstat
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from evaluate import load
import re
from rouge import Rouge
import pandas as pd
from medcat.cat import CAT

# small model
# cat = CAT.load_model_pack("models/umls_sm_pt2ch_533bab5115c6c2d6.zip")

bertscore = load("bertscore")

nltk.download('stopwords')
nltk.download('punkt')

class EvaluationMetrics:
    def __init__(self, model_name, typ="backward_vanilla", cat_model="models/umls_sm_pt2ch_533bab5115c6c2d6.zip",
                 current_explanation_file = "prompt_response_vanilla_1709427222.4795358.csv"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.cat_model = CAT.load_model_pack(cat_model)
        self.gcloud_folder = ""

        self.typ = typ
        
        self.current_explanation_file = current_explanation_file
        self.current_explanation_df = pd.read_csv(str(self.current_explanation_file))

        self.row_ids = self.current_explanation_df['row_id'].tolist()
        with open('diagnosis_reference_map.json', 'r') as json_file:
            self.diagnosis_reference_map = json.load(json_file)
        

    def setup(self):
        # drive.mount('/content/drive')
        return True

    def number_of_tokens(self, text):
        # Use Hugging Face tokenizer to get average number of tokens
        tokens = self.tokenizer(text, return_tensors="pt")['input_ids']
        return tokens.size(1)  # Assuming the tokenizer returns tokenized input as a tensor

    def bertscore_similarity(self, ref, hyp, max_chunk_len=512):
        scores = bertscore.compute(predictions=[hyp], references=[ref], lang="en")

        # Calculate average precision
        precision =  scores['precision'][0]
        return precision

    def get_ner_words(self, text, type="pretty_name"):
        # other person -> 'source_value'
        text_list = text.split(".")
        ner_ref_words = []
        for text in text_list:
            result = self.cat_model.get_entities(text)
            for entity in result['entities'].values():
                ner_ref_words.append(entity[type])
        return ner_ref_words

    def jaccard_score(self, hyp_words, ref_words):
        hyp_set = set(hyp_words)
        ref_set = set(ref_words)
        intersection = len(hyp_set.intersection(ref_set))
        union = len(hyp_set.union(ref_set))
        if union > 0:
            ans = intersection / union
        else:
            ans = 0
        return ans

    def jaccard_precision(self, hyp_words, ref_words):
        hyp_set = set(hyp_words)
        ref_set = set(ref_words)
        intersection = len(hyp_set.intersection(ref_set))
        return intersection / len(ref_set)

    def get_texts_synthetic(self, row_id, i):
        print("Synthetic: ")
        ground_truth = self.current_explanation_df[self.current_explanation_df['row_id'] == row_id]['gpt4_turbo_response'].values[0]
        prompt = self.current_explanation_df[self.current_explanation_df['row_id']==row_id][f'prompt_{i}'].values[0]
        response = self.current_explanation_df[self.current_explanation_df['row_id']==row_id][f'explanation_{i}'].values[0]
        return {'ground_truth': ground_truth, 'prompt': prompt, 'response': response}
    
    def combination_metrics(self, hyp, ref):
        hyp_words = self.get_ner_words(hyp, 'pretty_name')
        ref_words = self.get_ner_words(ref, 'pretty_name')
        jaccard_score = self.jaccard_score(hyp_words, ref_words)
        similarity = self.bertscore_similarity(hyp, ref)
        rouge_val = self.compute_rouge_n(hyp, ref)
        return {'jaccard_score': jaccard_score, 'bert_score': similarity, 'rouge_1': rouge_val['rouge-1']['p'], 'rouge_2': rouge_val['rouge-2']['p'], 'rouge_l': rouge_val['rouge-l']['p']}

    def compute_rouge_n(self, hypothesis, reference, n=1):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]
    
    def count_mask(self, text):
        return text.count("[MASK]")

    def evaluate(self):
        # Create lists to store results
        row_ids_list = []
        jaccard = {}
        bert_score_sim = {}
        rouge_1s = {}
        rouge_2s = {}
        rouge_ls = {}
        num_mask = {}
        num_tokens_list = {}
        num_med_ents = {}

        for i in range(5):
            jaccard[i] = []
            bert_score_sim[i] = []
            rouge_1s[i] = []
            rouge_2s[i] = []
            rouge_ls[i] = []
            num_mask[i] = []
            num_tokens_list[i] = []
            num_med_ents[i] = []

        for row_id in self.row_ids:
            row_ids_list.append(row_id)
            # ground_truth = texts["gpt4_turbo_response"].strip()

            for i in range(5):
                texts = self.get_texts_synthetic(row_id, i)
                response = texts[f"response"].strip()
                ground_truth = texts["ground_truth"].strip()

                num_tokens = self.number_of_tokens(response)
                c_mask = response.count("[MASK]")
                med_ents = self.get_ner_words(response)

                # # informativeness -> hype, reference
                cm = self.combination_metrics(response, ground_truth)
                jaccard[i].append(cm['jaccard_score'])
                bert_score_sim[i].append(cm['bert_score'])
                rouge_1s[i].append(cm['rouge_1'])
                rouge_2s[i].append(cm['rouge_2'])
                rouge_ls[i].append(cm['rouge_l'])

                # Add results to lists
                
                num_tokens_list[i].append(num_tokens)
                num_mask[i].append(c_mask)
                num_med_ents[i].append(len(med_ents))

                print(f"row_id: {row_id} masking_level: {i} count_mask: {c_mask} num_med_ents: {len(med_ents)}")
                print(f"cm: {cm}")
                print("response: ", response)
                print("#####################################################################################################################")


        # Create a DataFrame
        data = {
            'row_id': row_ids_list
        }

        for i in range(5):
            data[f'jaccard_{i}'] = jaccard[i]
            data[f'bert_score_{i}'] = bert_score_sim[i]
            data[f'rouge_1s_{i}'] = rouge_1s[i]
            data[f'rouge_2s_{i}'] = rouge_2s[i]
            data[f'rouge_ls_{i}'] = rouge_ls[i]
            data[f'num_token_{i}'] = num_tokens_list[i]
            data[f'num_mask_{i}'] = num_mask[i]
            data[f"num_med_ents_{i}"] = num_med_ents[i]
        
        # Save the DataFrame to a CSV file
        result_df = pd.DataFrame(data)
        result_df.to_csv("new_data/evaluated_response/" + f'evaluation_{self.typ}_results_new_{time.time()}.csv', index=False)
        
        print("average: ", result_df.mean())
        

                
print("############################################################################################################")

# Example Usage:
model_name = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"

# make sure that the model is available in the following folder
cat_model = "models/umls_sm_pt2ch_533bab5115c6c2d6.zip"

file_map = {"gpt3.5_turbo": "final_data/masked/masked_response_gpt-3.5-turbo.csv",
            "biomistral": "final_data/masked/masked_texts_biomistral.csv",
            "mistral": "final_data/masked/masked_texts_mistral.csv",
            "clinical_panda": "final_data/masked/masked_texts_clinical_panda.csv"
            } 

for typ, current_explanation_file in file_map.items():
    evaluator = EvaluationMetrics(model_name, typ, cat_model, current_explanation_file)
    evaluator.evaluate()
    print("############################################################################################################")

