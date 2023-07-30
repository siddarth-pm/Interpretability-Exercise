import transformers
from tabulate import tabulate
import tqdm as tqdm
import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
import numpy as np



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2")
from datasets import load_dataset

dataset = load_dataset("sst")

batch_size = 16


dataloader = DataLoader(eval_ds, batch_size=batch_size)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")

from transformers import TextClassificationPipeline
classifier = TextClassificationPipeline(model=model,
                                        tokenizer=tokenizer,
                                        device=device)


# Explanation methods:

# Gradient
def compute_gradient_saliency(mdel, dl):
  mdel.enable_input_require_grads()
  gradients = []
  word_embeds = mdel.get_input_embeddings()
  for batch in tqdm(dl):
    for idx in range(len(batch['sentence'])):
      sentence = batch['sentence'][idx]
      x = tokenizer(sentence, return_tensors="pt")["input_ids"]
      input_embeds = word_embeds(x)
      input_embeds.retain_grad()
      y = mdel(inputs_embeds = input_embeds)
      y["logits"][0][0].backward()
      g = (input_embeds.grad)[0]
      cur_gradients = {}
      tkns = []
      for i in range(len(x[0])):
        decoded = tokenizer.decode(x[0][i])
        cur_gradients[decoded] = (np.linalg.norm(g[i]), i)
        tkns.append(decoded)
      cur_gradients = sorted(cur_gradients.items(), key=lambda x: abs(x[1][0]), reverse=True)
      gradients.append((cur_gradients, tkns))
  mdel.disable_input_require_grads()
  return gradients


# Occlusion
def compute_occlusion_saliency(clsifier, dl):
  occlusions = []
  for batch in tqdm(dl):
    with torch.no_grad():
      batch_pre_y = clsifier(batch['sentence'])
    for idx in range(len(batch['sentence'])):
      tkns = batch['tokens'][idx].split('|')
      cur_occlusions = {}
      pre_y = batch_pre_y[idx]
      for i in range(len(tkns)):
        word = tkns[i]
        tkns[i] = ''
        phrase = ' '.join(tkns)
        with torch.no_grad():
          y = clsifier(phrase)
        if(pre_y['label'] == y[0]['label']):
          cur_occlusions[word] = (pre_y['score'] - y[0]['score'], i)
        else:
          cur_occlusions[word] = (abs(pre_y['score']) + abs(y[0]['score']), i)
        tkns[i] = word

      cur_occlusions = sorted(cur_occlusions.items(), key=lambda x: abs(x[1][0]), reverse=True)
      occlusions.append((cur_occlusions, tkns))
  return occlusions


# LIME
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
from lime.lime_text import IndexedString
import numpy as np

# from timeit import default_timer as timer

def predict_proba(sentences):
  probs = np.zeros((len(sentences), 2), dtype=float)
  preds = classifier(sentences)
  for i in range(len(sentences)):
    lb = int(preds[i]['label'][-1])
    probs[i, lb] = preds[i]['score']
    probs[i, 1 - lb] = 1 - probs[i, lb]
  return probs


def compute_lime_saliency(dl, class_names):
  explanations = []
  explainer = LimeTextExplainer(class_names=class_names, bow=False)
  for batch in tqdm(dl):
    for idx in range(len(batch['sentence'])):
      sent = batch['sentence'][idx]
      indexed_string = IndexedString(sent, bow=False)
      with torch.no_grad():
        exp = explainer.explain_instance(sent, predict_proba, num_features=indexed_string.num_words(), num_samples=1000)
      exp = exp.as_list()
      cur_limes = {}
      tkns = []
      for i in range(len(exp)):
        tkns.append(exp[i][0])
      for i in range(len(exp)):
        cur_limes[exp[i][0]] = (exp[i][1], i)
      cur_limes = sorted(cur_limes.items(), key=lambda x: abs(x[1][0]), reverse=True)
      explanations.append((cur_limes, tkns))
  return explanations


# Metrics

# Comprehensiveness
def calculate_comprehensiveness(explanations):
  ret_list = []

  for i in range(len(explanations)):
    ret = 0
    explanation = explanations[i][0]
    tkns = explanations[i][1][:]
    phrase = ' '.join(tkns)
    with torch.no_grad():
      pre_y = classifier(phrase)
    for j in range(len(explanation)):
      index = explanation[j][1][1]
      tkns[index] = ''
      phrase = ' '.join(tkns)
      with torch.no_grad():
        y = classifier(phrase)
      if(pre_y[0]['label'] == y[0]['label']):
        ret += (pre_y[0]['score'] - y[0]['score'])
      else:
        ret += abs(pre_y[0]['score']) + abs(y[0]['score'])
    ret = ret / (len(explanation) + 1)
    ret_list.append(ret)

  return ret_list

# Sufficiency
def calculate_sufficiency(explanations):
  ret_list = []
  for i in tqdm(range(len(explanations))):
    ret = 0
    explanation = explanations[i][0]
    tkns = explanations[i][1][:]
    masked_tkns = ['' for token in tkns]
    phrase = ' '.join(tkns)
    with torch.no_grad():
      pre_y = classifier(phrase)
    for j in range(len(explanation)):
      index = explanation[j][1][1]
      masked_tkns[index] = tkns[index]
      phrase = ' '.join(masked_tkns)
      with torch.no_grad():
        y = classifier(phrase)
      if(pre_y[0]['label'] == y[0]['label']):
        ret += (pre_y[0]['score'] - y[0]['score'])
      else:
        ret += abs(pre_y[0]['score']) + abs(y[0]['score'])
    ret = ret / (len(explanation) + 1)
    ret_list.append(ret)


  return ret_list


# DF_MIT
def calculate_avg_DF_MIT(explanations):
  flipped = 0
  total = 0
  for i in range(len(explanations)):
    explanation = explanations[i][0]
    tkns = explanations[i][1][:]
    phrase = ' '.join(tkns)
    with torch.no_grad():
      pre_y = classifier(phrase)
    pre_y_label = int(pre_y[0]['label'][-1])
    if(len(explanation) == 0):
      continue
    index = explanation[0][1][1]
    tkns[index] = ''
    phrase = ' '.join(tkns)
    with torch.no_grad():
      y = classifier(phrase)
    y_label = int(y[0]['label'][-1])
    if y_label != pre_y_label:
      flipped += 1
    total += 1

  return flipped/total


# DF_Frac
def calculate_avg_DF_Frac(explanations):
  ret_list = []
  
  for i in range(len(explanations)):
    explanation = explanations[i][0]
    tkns = explanations[i][1][:]
    phrase = ' '.join(tkns)
    with torch.no_grad():
      pre_y = classifier(phrase)
    pre_y_label = int(pre_y[0]['label'][-1])
    num_taken = len(explanation) + 1
    for j in range(len(explanation)):
      index = explanation[j][1][1]
      tkns[index] = ''
      phrase = ' '.join(tkns)
      with torch.no_grad():
        y = classifier(phrase)
      y_label = int(y[0]['label'][-1])
      if y_label != pre_y_label:
        num_taken = j + 1
        break
    ret_list.append(num_taken/(len(explanation) + 1))

  return calculate_avg(ret_list)

def calculate_avg(cmp_list):
  return sum(cmp_list) / len(cmp_list)


def gradient_saliency_metrics():
    gradient_explanations = compute_gradient_saliency(classifier, dataloader)
    print("Found explanations.")
    gradient_comprehensivness = calculate_comprehensiveness(gradient_explanations)
    avg_gradient_comprehensiveness = calculate_avg(gradient_comprehensivness)
    print("Comprehensiveness done.")
    gradient_sufficiency = calculate_sufficiency(gradient_explanations)
    avg_gradient_sufficiency = calculate_avg(gradient_sufficiency)
    print("Sufficiency done.")
    gradient_df_mit = calculate_avg_DF_MIT(gradient_explanations)
    print("DF MIT Done.")
    gradient_df_frac = calculate_avg_DF_Frac(gradient_explanations)
    print("DF Frac Done.")

    print("Gradient Sufficiency: ", avg_gradient_sufficiency)
    print("Gradient Comprehensiveness: ", avg_gradient_comprehensiveness)
    print("Gradient DF_MIT: ", gradient_df_mit)
    print("Gradient DF_Frac: ", gradient_df_frac)

    return ["Gradient", avg_gradient_sufficiency, avg_gradient_comprehensiveness, gradient_df_mit, gradient_df_frac]

def occlusion_saliency_metrics():
    occlusion_explanations = compute_occlusion_saliency(classifier, dataloader)
    print("Found explanations.")
    occlusion_comprehensivness = calculate_comprehensiveness(occlusion_explanations)
    avg_occlusion_comprehensiveness = calculate_avg(occlusion_comprehensivness)
    print("Comprehensiveness done.")
    occlusion_sufficiency = calculate_sufficiency(occlusion_explanations)
    avg_occlusion_sufficiency = calculate_avg(occlusion_sufficiency)
    print("Sufficiency done.")
    occlusion_df_mit = calculate_avg_DF_MIT(occlusion_explanations)
    print("DF MIT Done.")
    occlusion_df_frac = calculate_avg_DF_Frac(occlusion_explanations)
    print("DF Frac Done.")

    print("Occlusion Sufficiency: ", avg_occlusion_sufficiency)
    print("Occlusion Comprehensiveness: ", avg_occlusion_comprehensiveness)
    print("Occlusion DF_MIT: ", occlusion_df_mit)
    print("Occlusion DF_Frac: ", occlusion_df_frac)
    return ["Occlusion", avg_occlusion_sufficiency, avg_occlusion_comprehensiveness, occlusion_df_mit, occlusion_df_frac]

def LIME_saliency_metrics():
    lime_explanations = compute_lime_saliency(dataloader, [0, 1])
    print("Found explanations.")
    lime_comprehensivness = calculate_comprehensiveness(lime_explanations)
    avg_lime_comprehensiveness = calculate_avg(lime_comprehensivness)
    print("Comprehensiveness done.")
    lime_sufficiency = calculate_sufficiency(lime_explanations)
    avg_lime_sufficiency = calculate_avg(lime_sufficiency)
    print("Sufficiency done.")
    lime_df_mit = calculate_avg_DF_MIT(lime_explanations)
    print("DF MIT Done.")
    lime_df_frac = calculate_avg_DF_Frac(lime_explanations)
    print("DF Frac Done.")

    print("LIME Sufficiency: ", avg_lime_sufficiency)
    print("LIME Comprehensiveness: ", avg_lime_comprehensiveness)
    print("LIME DF_MIT: ", lime_df_mit)
    print("LIME DF_Frac: ", lime_df_frac)
    return ["LIME", avg_lime_sufficiency, avg_lime_comprehensiveness, lime_df_mit, lime_df_frac]

if __name__ == "__main__":
  print("Starting evaluation process.")
  print("----------Gradients----------")
  gradient_metrics = gradient_saliency_metrics()

  print("----------Occlusion----------")
  occlusion_metrics = occlusion_saliency_metrics()
  print("-------------LIME------------")
  lime_metrics = LIME_saliency_metrics()
  names = ["Explanation", "Sufficiency", "Comprehensiveness", "DF_MIT", "DF_Frac"]
  table = tabulate([names, gradient_metrics, occlusion_metrics, lime_metrics])
  print("-------------Summary------------")
  print(table)
