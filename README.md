# AI-PROJECT - 282491
Project for the course of *Artificial Intelligence and Machine Learning* taught by Prof. Italiano, for the BSc in Management and Computer Science.
Group components are Francesca Romana Sanna, Leonardo Azzi and Davide Pisano.

## Introduction
Artificial intelligence is a field that has always fascinated the three of us, and since we were already studying and learning on our own, we decided to push the limits of our project and consider a different approach: we present to you a work on *text classification* with **Large Language Models (LLMs)**, leveraging the power of their *transformers architecture* with an innovative technique.


## Methods
Instead of using simple/traditional models like the ones we studied in class - such as linear regression, logistic regression, SVM - we are taking a leap forward and applying a *twist* to LLMs; rather than modifying the underlying architecture, we want the model to output a *structured prediction* that we can then extract and apply to classify the text. We have chosen **JSON** for this task, even though other languages such as *YAML* would work fine in this context.

### Why did we make these decisions?
We have chosen a **structured output** as it is way easier to implement in any system, regardless of programming languages or interfaces. **JSON** is a very popular format, and it is also very easy to parse and extract the information we need. As we said before, we have also considered YAML, but we decided to stick with the first one as it is more widely used and also easier to read and understand, despite the *higher token usage* (which is not a problem in our case, as we will show later on).


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