import pandas as pd
import json
from collections import Counter
from ast import literal_eval
from openai import OpenAI
import time

class ClinicalAssistant:
    def __init__(self):
        self.client = OpenAI()
        with open("release_conditions.json", "r") as f:
            self.data_pathology = json.load(f)

        with open("release_evidences.json", "r") as f:
            self.data_evidences = json.load(f)

    def differential_dia(self, pathology_probabilities):
        resp = [p[0] for p in pathology_probabilities]
        return resp

    def generate_medical_statement(self, qa_pair):
        evidence_type = "antecedent" if qa_pair['is_antecedent'] else "symptom"
        statement = f"The patient answered {qa_pair['answer']} for the question \"{qa_pair['question']}\" and this evidence is a medical {evidence_type}."
        return statement
    
    def get_qa(self, evidences):
        qa_pairs = []
        for evidence in evidences:
            ev_list = evidence.split("_@_")
            qa_pair = {}
            evidence_no = ev_list[0]
            if len(ev_list) > 1:
                answer_no = ev_list[1]
                val = self.data_evidences[evidence_no]['value_meaning']
                if val is not None:
                    ans = val.get(answer_no)
                    if ans is not None:
                        answer = ans.get('en')
            else:
                answer = True if self.data_evidences[evidence_no]['default_value'] == 0 else False

            qa_pair['question'] = self.data_evidences[evidence_no].get('question_en')
            qa_pair['is_antecedent'] = bool(self.data_evidences[evidence_no].get('is_antecedent'))
            qa_pair['answer'] = answer
            qa_pairs.append(qa_pair)
        return qa_pairs

    def get_prompt(self, row):
        differential_diagnosis_desc = self.differential_dia(row['DIFFERENTIAL_DIAGNOSIS'])
        qa_pairs = self.get_qa(row['EVIDENCES'])
        init_qa = self.get_qa([row['INITIAL_EVIDENCE']])
        text = f"""
        For the following information of a patient, develop a clinical report with the following sections: Chief Complaint (CC), History of Present Illness (HPI), Past Medical History (PMH), Medications and Allergies, Physical Examination (PE):
        
        Patient Details are as follows:
        Age: {row['AGE']}
        Sex: {row['SEX']}
        Initial Evidence: {init_qa}

        Quesion answers:
        The following are the question and answers from the patient: 
        """
        for qa_pair in qa_pairs:
            statement = self.generate_medical_statement(qa_pair)
            text = text + "\n" + statement + "\n"
        text = text + f"""\n
        For the above information of a patient generate a clinical report with the following template:
        Patient Details: Age and Sex
        Chief Complaint: Is obtained from the following question and answer: {init_qa}
        History of Present Illness: Briefly describe:
            Symptoms: Onset, duration, severity, aggravating/alleviating factors
            Relevant history: Leading events, prior treatments
        Past Medical History: List significant conditions (e.g., diabetes, allergies)
        Medications: List current medications (name, dosage)
        Physical Exam:
            Vitals: Briefly summarize vitals (e.g., T: 98.6, HR: 100, BP: 140/90)
            General: Briefly describe appearance (e.g., awake/alert/oriented)
            Briefly mention relevant positive findings related to the chief complaint.
        Assessment: Briefly summarize key findings and formulate a preliminary assessment (e.g., suspected pneumonia).
        Plan: Outline initial plan (e.g., chest X-ray, antibiotics).
        """
        return text

    def get_response(self, prompt, model = "gpt-4", max_tokens=512):
        response = self.client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant helping to create a comprehensive clinical report with the following sections: Chief Complaint (CC), History of Present Illness (HPI), Past Medical History (PMH), Medications and Allergies, Physical Examination (PE)."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    def process_test_data(self, df_test, model = "gpt-4"):
        responses = []
        prompts = []
        for idx, row in df_test.iterrows():
            prompt = self.get_prompt(row)
            prompts.append(prompt)
            print("idx: ", idx)
            resp = self.get_response(prompt, model)
            responses.append(resp)
            print("response: ", resp)

        df_test['gpt4_response'] = responses
        df_test['prompt'] = prompts

        return df_test

if __name__ == "__main__":
    seed = 42
    limit = 100
    model = "gpt-4-1106-preview"
    df = pd.read_csv('final_data/release_test_patients.csv')
    # df_pool = df.copy()
    df_test = df.sample(n=limit, random_state=seed)  # Select all rows

    df_pool =  df_test
    # Remove samples that are already in df_test
    # df_pool = df_pool.drop(df_test.index, errors='ignore')
    
    # df_pool = df_pool.sample(n=1000, random_state=seed)
    # # first 100 ones
    # df_pool = df_pool[100:1000]

    df_pool['EVIDENCES'] = df_pool['EVIDENCES'].apply(literal_eval)
    df_pool['DIFFERENTIAL_DIAGNOSIS'] = df_pool['DIFFERENTIAL_DIAGNOSIS'].apply(literal_eval)
    clinical_assistant = ClinicalAssistant()
    df_pool = clinical_assistant.process_test_data(df_pool)
    df_pool.to_csv(f'{model}_test_clinical_notes_{time.time()}__fixed__responses.csv')
