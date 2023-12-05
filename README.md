# AI-PROJECT - 282491
Project for the course of *Artificial Intelligence and Machine Learning* taught by Prof. Italiano, for the BSc in Management and Computer Science.
Group components are Francesca Romana Sanna, Leonardo Azzi and Davide Pisano.

## Introduction
Artificial intelligence is a field that has always fascinated the three of us, and since we were already studying and learning on our own, we decided to push the limits of our project and consider a different approach: we present to you a work on *text classification* with **Large Language Models (LLMs)**, leveraging the power of their *transformers architecture* with an innovative technique. 
Zephyr-7B is the model we have used, which is a fine-tuned version of Mistral-7B, an outstanding pre-trained model that excels in NLP tasks.

### Why LLMs?
LLMs are a rapidly growing and evolving field, we are witnessing a revolution in the NLP world, and we wanted to be part of it. Although they are not the best models for this kind of task, as they could easily be considered "overkill", we wanted to see how they would perform and how we could improve them to make them more suitable, while also learning more about them, their inner workings and research on structured-formatted outputs.

## Methods
So instead of using simple/traditional models like the ones we studied in class - such as linear regression, logistic regression, SVM - we are taking a leap forward and applying a *twist* to LLMs; rather than modifying the underlying architecture, we want the model to output a *structured prediction* that we can then extract and apply to classify the text. We have chosen **JSON** for this task, even though other languages such as *YAML* would work fine in this context.

Quick example:
```
"Is this message fraudolent?"

"Click here for free money!

"{ "is_fraudolent": true }"
```
Extract the information in any programming language and you are good to go.

### Why JSON?
We have chosen a **structured output** as it is way easier to implement in any system, regardless of programming languages or interfaces, and it allows the task to be extended with minimal effort. **JSON** is a very popular format, and it is also very easy to parse and extract the information we need. As we said before, we have also considered YAML, but we decided to stick with the first one as it is more widely used and also easier to read and understand for us, despite the *higher token usage* (which is not a problem in our case, as we will show you later on).

### Why did we choose this approach?
We wanted to gain experience and learn more about these models, how to use them, fine-tune them and obtain responses (structured) that could be used in programming interfaces (hence why JSON was chosen). We also wanted to see how they would perform on a task that is not their main purpose, but that they should be really good at handling by nature, and how we could improve them to make them more suitable for this kind of task. The learning path is the primary reason behind our choice.

### Why this model?
**Zephyr-7B** is a fine-tuned version of *Mistral-7B*, it has been fine-tuned on GPT-4 enhanced datasets (*Ultrachat* + many others) and is a better performing LLM than its fundational model Mistral. The *7B parameters* are (from an LLM prospective) a good balance between performance, speed and memory usage, as the 3B models were too underperforming and realistically unusable, while the 13B models were too big and memory intensive for our purposes and for fine-tuning on our available hardware. Additionally, 7B models are more commonly used, especially in real world applications, and there are way more versions and resources available for them.\
We also took into consideration *LLAMA 2* based models (such as *Orca 2*), or *YI* based models, but Mistral scored better (for 7B models) in our specific context, and we wanted to stick with it.
So we have chosen a model which is not too computationally expensive but is, on the other hand, powerful enough, meaningful in terms of learning path, and that we could fine-tune on our hardware.


### How can you recreate our work?
We picked the 3.10 version of Python to count both on recent support and more stability across different machines. As for the models, we provide below necessary and recommended steps to load and run them.

**Requirements:** 
- install the dependencies listed in the `requirements.txt` file with your selected package manager (e.g. `pip install -r requirements.txt`)

**Recommended use:**
- having `pip` and `pipenv` already installed
- create a folder in the root named `.venv` and run `pipenv install` in the terminal to install all the various dependencies from the `Pipfile` and `Pipfile.lock` files
- run `pipenv shell` to activate the virtual environment

**Notes:**
as you can see there are two other GitHub repositories linked in the `requirements.txt` file, designated to simplify our work; one (*LLaMA-Factory*) regarding the fine tuning of the LLMs, and the other (*llama.cpp*) regarding the export of the model to a more suitable format - *GGUF*.

**Extra:**
if you want to get the full experience, we uploaded the model with different formats (*Pytorch, GGUF 8/16-bit LoRA quantized*) on HuggingFace. In this way, you could easily run inference on a CPU, exploiting the complete interface provided by services like *LMStudio* (strongly recommended).



### 
We tested zephyr on the dataset, got bad results for both JSON output and task performance then fine-tuned and quantized, compared performance and got massive improvements on both JSON output (100% on test) and on task performance.

Shrink model + fine tuned = more specialized LLM that also excels in JSON outputs and Message text classification, while also behaving like a normal LLM on other tasks. This mitigates the issue of the model being too computationally expensive, it still is, but less.

We converted to GGUF for easier CPU inference, making it computationaly cheaper and more accessible.






Describe your proposed ideas (e.g., features, algorithm(s),
training overview, design choices, etc.) and your environment so that:
• A reader can understand why you made your design decisions and the
reasons behind any other choice related to the project!!!!!!!!!!!!!!!!!!!!!!!!
• A reader should be able to recreate your environment (e.g., conda list,
conda envexport, etc.)XXXXXXXXXXXXXXXXXXXXXX
• It may help to include a figure illustrating your ideas, e.g., a flowchart
illustrating the steps in your machine learning system(s)




## Instructions
Perform an Explanatory data analysis (EDA) with visualization using the entire dataset.. 
• Preprocess the dataset (impute missing values, encode categorical features with one-hot 
encoding). Your goal is to estimate whether an SMS is fraudulent 
• Define whether this is a regression, classification or clustering problem, explain why and 
choose your model design accordingly. Test at least 3 different models. First, create a 
validation set from the training set to analyze the behaviour with the default 
hyperparameters. Then use cross-validation to find the best set of hyperparameters. You 
must describe every hyperparameter tuned (the more, the better) 
• Select the best architecture using the right metric 
• Compute the performances of the test set 
• Explain your results 

## Title and Team members

### [Section 1] Introduction – Briefly describe your project

### [Section 2] Methods – Describe your proposed ideas (e.g., features, algorithm(s),
training overview, design choices, etc.) and your environment so that:
• A reader can understand why you made your design decisions and the
reasons behind any other choice related to the project
• A reader should be able to recreate your environment (e.g., conda list,
conda envexport, etc.)
• It may help to include a figure illustrating your ideas, e.g., a flowchart
illustrating the steps in your machine learning system(s)

### [Section 3] Experimental Design – Describe any experiments you conducted to
demonstrate/validate the target contribution(s) of your project; indicate the
following for each experiment:
• The main purpose: 1-2 sentence high-level explanation
• Baseline(s): describe the method(s) that you used to compare your work
to
• Evaluation Metrics(s): which ones did you use and why?

### [Section 4] Results – Describe the following:
• Main finding(s): report your final results and what you might conclude
from your work
• Include at least one placeholder figure and/or table for communicating
your findings
• All the figures containing results should be generated from the code.

### [Section 5] Conclusions – List some concluding remarks. In particular:
• Summarize in one paragraph the take-away point from your work.
• Include one paragraph to explain what questions may not be fully
answered by your work as well as natural next steps for this direction of
future work