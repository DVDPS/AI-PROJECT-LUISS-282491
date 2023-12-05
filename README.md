# AI-PROJECT - 282491
Project for the course of *Artificial Intelligence and Machine Learning* taught by Prof. Italiano, for the BSc in Management and Computer Science.
Group components are Francesca Romana Sanna, Leonardo Azzi and Davide Pisano.

## Section 1 - Introduction
Artificial intelligence is a field that has always fascinated the three of us, and since we were already studying and learning on our own, we decided to push the limits of our project and consider a different approach: we present to you a work on *text classification* with **Large Language Models (LLMs)**, leveraging the power of their *transformers architecture* with an innovative technique. 

We picked the model **Zephyr-7B**, which is a fine-tuned version of **Mistral-7B**, an outstanding pre-trained model that excels in *NLP tasks*. We *evaluated* base-model performance, then *fine-tuned* it on our dataset, *quantized* it and *compared* the results. We also converted the model to a more suitable format for *CPU inference*, and we have uploaded it on **HuggingFace** for easier use.

### Why LLMs?
LLMs are a rapidly growing and evolving field, we are witnessing a revolution in the NLP world, and we wanted to be part of it. Although they are not the best models for this kind of task, as they could easily be considered "overkill", we wanted to see how they would perform and how we could improve them to make them more suitable, while also learning more about them, their inner workings and research on structured-formatted outputs.

## Section 2 - Methods
So instead of using simple/traditional models like the ones we studied in class - such as linear regression, logistic regression, SVM - we are taking a leap forward and applying a *twist* to LLMs; rather than modifying the underlying architecture, we want the model to output a *structured prediction* that we can then extract and apply to classify the text. We have chosen **JSON** for this task, even though other languages such as *YAML* would work fine in this context.

Quick example:
```
"Is this message fraudulent?"

"Click here for free money!

"{ "is_fraudulent": true }"
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
as you can see there are two other GitHub repositories linked in our `requirements`, picked to simplify our work; one (*LLaMA-Factory*) regarding the fine-tuning of the LLMs, and the other (*llama.cpp*) regarding the export of the model to a more suitable format - *GGUF* - and use of quantization techniques.

**Extra:**
if you want to get the full experience, we uploaded the model with different formats (*Pytorch, GGUF 8/16-bit LoRA quantized*) on HuggingFace. In this way, you could easily run inference on a CPU, exploiting the complete interface provided by services like *LMStudio* (strongly recommended).

## Section 3 - Experimental Design

### Main Purpose and Parameters
The main purpose of this project is to evaluate fine-tuning effectiveness on small models in obtaining **structured responses** and for **task-specific performance**.

For fine-tuning and "advanced" parameters we used defaults or suggested/best-practice values, as we were more interested in evaluating the model's performance and the effects of fine-tuning on it, rather than the effects of hyperparameters.

Below we briefly discuss the main parameters and hyperparameters we used throughout the project.

#### Model Hyperparameters
- `temperature`: 0.3 - As it is more towards a deterministic output, we want to avoid randomness in the response.
- `max_tokens`: 128 - We want to limit the tokens generated, as only a few are needed for our use case. This avoids the model generating unnecessary tokens and wasting time.

#### Instructions
- Fine-Tuning Instructions: `"Determine if the following SMS message is fraudulent. Output the response in JSON with the key "is_fraudulent" and the bool value."` - We want to limit the token usage in the instructions, and teach the model to pair these with the output we want, essentially strenghtening the connection between the key "JSON" and the structured output, as well as the task itself.

- Base Model Instructions: 

```You are excellent message moderator, expert in detecting fraudulent messages.

You will be given "Messages" and your job is to predict if a message is fraudulent or not.

You ONLY respond FOLLOWING this json schema:

{
    "is_fraudulent": {
        "type": "boolean",
        "description": "Whether the message is predicted to be fraudulent."
    }
}

You MUST ONLY RESPOND with a boolean value in JSON. Either true or false in JSON. NO EXPLANATIONS OR COMMENTS.

Example of valid responses:
{
    "is_fraudulent": true
}
or 
{
    "is_fraudulent": false
}
```

The base model required substantially more instructions to be able to perform the task with a decent performance/accuracy. From our tests, it struggled to maintain consistency in the JSON, and it was not able to perform with a good accuracy, probably due to the contaminated instructions and attention focus on the output, rather than the task. This is covered in-depth in the next section.


### Model fine-tuning
For fine-tuning the model we needed to:
1. Convert the data in a suitable format
2. Create a training set and a validation set
3. Fine-tune the model


#### 1. Data format
To use the dataset (`sms.csv`), we had to convert the labels in a meaningful format for the model. Since we wanted to use JSON, we converted the fraudolence lables (*0* and *1*) to a boolean value (*true* and *false*) in JSON with the key `"is_fraudulent"`.
We did this following the **Alpaca Format** (LLM format), here is a snippet:
```json
[
  {
    "instruction": "Determine if the following SMS message is fraudulent. Output the response in JSON with the key \"is_fraudulent\" and the bool value.",
    "input": "Cool. Do you like swimming? I have a pool and jacuzzi at my house.",
    "output": "{\"is_fraudulent\": false}"
  },
  {
    "instruction": "Determine if the following SMS message is fraudulent. Output the response in JSON with the key \"is_fraudulent\" and the bool value.",
    "input": "URGENT! You have won a 1 week FREE membership in our \u00a3100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
    "output": "{\"is_fraudulent\": true}"
  },
  ...
]
```

#### 2. Training and validation set
We normally split the dataset in a *training set* and a *validation set*, with a *80/20 ratio*, making sure the ratio of "fraudulent" and "non-fraudulent" messages is the same in both sets. However...

Through our EDA we discovered that the dataset was *strongly unbalanced* towards the non-fraudulent class. Normally we would have used a stratified split, but in this case we decided to keep the imbalance for a specific reason.
The base model had a really high recall and a really low precision, which means that it was able to detect most (actually all) of the fraudulent messages, but it also classified many non-fraudulent messages as fraudulent (giving *false positives*). This is of course a major issue, and we wanted to see if fine-tuning could mitigate it; hence why we decided to keep the imbalance in the dataset and try to improve the model precision, while also keeping a good recall.

#### 3. Fine-tuning
We fine-tuned the model using **LLaMA-Factory**, a tool that allows for easy fine-tuning of LLMs with a simple web interface.
This not only cuts down the time needed, as it reduces technical complexity, but it also makes our environment more accessible and reproducible.

Speaking of hardware, we went through the whole process on an *NVIDIA RTX 4090 with 24GB of VRAM*. We provide the screenshots below, to give you a more clear overview.
1. Loaded "Zephyr-7B".
2. Loaded the dataset.
3. Set the parameters.
4. Started fine-tuning!

![Fine-tuning](images\Zephyr-Base-Fine-Tuning.png)

(Results sneak peak: *"train_loss" = 0.03747267646436822*)

### Quantization
As we previously dicussed, LLMs require a lot of computational power and memory, which is a big issue especially for "relatively simple tasks" like this one. To mitigate this issue, we decided to quantize the model, which means reducing the precision of the weights and activations, and therefore reducing the memory usage and computational power needed to run the model.

To do so, we used **LLAMA.cpp**, a general tool for LLM inference which also facilitates quantization. We first converted the model to the GGUF format, which is a format that allows for easier inference on CPUs (additional "computation friendliness"), and then we created 2 quantized versions of the model:
- 8-bit (Q_8 QLoRA) - [Available here](https://huggingface.co/SimplyLeo/Zephyr-Fraudulence-Detector/blob/main/zehpyr-fraudulence-detector-q8_0.gguf)
- 16-bit (fp16) - [Available here](https://huggingface.co/SimplyLeo/Zephyr-Fraudulence-Detector/blob/main/Zephyr_Fraudolence_detector_full.gguf)

We then uploaded the models on our [HuggingFace Repo](https://huggingface.co/SimplyLeo/Zephyr-Fraudulence-Detector/tree/main) to make them more accessible and easy to use.

Quantization, GGUF conversion and HuggingFace upload allowed us to make our model more accessible and easier to use, as it can now be run on CPUs (with layers offload to the GPU if possible) and with many LLM services, such as [LMStudio](https://lmstudio.ai), that we strongly recommend.

> Note: to get our model running on LM Studio and other services (assuming HuggingFace support), you simply need to search for the model name (Zephyr-Fraudulence-Detector) or Repo (SimplyLeo/Zephyr-Fraudulence-Detector) and select the model you want to use. You can then run inference on the model, and you can also fine-tune it on your own dataset, if you want to (Not recommended). 


![Search Model](images\LM-Studio-Model-Search.png)

## Section 4 - Results

Now **we finally did it!**\
On the first instance of the base model, we observed that the scores and results were not really optimal, both concerning the task - keeping an eye on the performance metrics - and the JSON output - where we had a lot of invalid structures and content.\
After the fine-tuning phase, we obtained a model that is able to classify messages as fraudolent or not, and we can also extract the information we need from the JSON output.\
Some stats we recorded:
| **Metric**                       | **Base model (Zephyr-7b)** | **Fine-tuned** |
| :---                             |    :----:                  |     :---:      |
| *Valid responses*                | 71.69%                     | 1115 - 100%    |
| *Invalid JSON structures*        | 28.20%                     | 0 - 0%         |
| *Invalid JSON content*           | 0.11%                      | 0 - 0%         |
| *Accuracy for valid responses*   | 17.65%                     | 99%            |
| *Precision for valid responses*  | 17.65%                     | 99%            |
| *Recall for valid responses*     | 100%                       | 97%            |
| *F1 score for valid responses*   | 30%                        | 98%            |


These are more than impressive following the previous evaluation, where the model scored okay on the JSON output, but achieved poorly on the metrics, with the exception of the *recall* being perfect since it actually guessed right all the fraudolent messages.\
Following a more in-depth analysis, we can see that the model is able to classify correctly messages that are not in the training set, and that it is able to format the information in JSON output, which is the main goal of our project.\
We accomplished an improvement in the **JSON output and compliance**: having *zero* instances of *invalid* structures or content, this now means that our model is consistently generating correct outputs, which is a fundamental step for the task.\
Moreover, we managed to take a big step forward with the **accuracy, precision, recall and F1 score**: going from a 17.65% to a 99% *accuracy* indicated the model is *almost perfectly doing its job* at identifying fraudulent messages, while the high *F1 score* - mixing precision and recall in one - points out a great balance overall.\
Here we provide a second analysis of the outcomes, with some other key metrics such as the **BLEU and ROUGE scores** used for *NLP evaluation*. They add up to the positive performances we have seen so far, although we won't dive into them as they are too far apart from what we studied during our course.
| **Metric**            | **Base model (Zephyr-7b)** | **Fine-tuned** |
| :---                  |    :----:                  |     :---:      |
| *BLEU-4*              | 21.95%                     | 99.85%         |
| *ROUGE-1*             | 43.65%                     | 99.92%         |
| *ROUGE-2*             | 38.07%                     | 99.84%         |
| *ROUGE-L*             | 40.03%                     | 99.93%         |


All these excellent results can be attributed to the work done with the *fine-tuning*, transforming the base model into a faster, more efficient and more specialized one, which is able to perform significantly better on the task while needing less time and computational power, while also excelling in generating a structured JSON output.


## Section 5 - Conclusions
Considering all this, we showed that the fine-tuned model vastly outperforms the base model on every aspect we measured - again, accuracy, precision, recall, F1 score, BLEU and ROUGE scores, output accuracy and efficiency. Overall, we accomplished basically perfect text classification, managing to have the model predict correctly all messages, both in the test set and made up by us, with a satisfactory structured output.

We are aware that using an LLM for this kind of task might not the best choice (at least not in this state), as it requires a significant amount of computational power and memory (considering that simpler models can still achieve quite effective results) and it is not ready for real world applications. However, with this project we managed to learn a lot about using LLMs, fine-tuning, quantization, different formats and, most importantly, tweaking the model to get structured outputs, while also getting excellent score on the message classification.

Additionally, the final model is not limited by the training on the task we imposed, and changing the instructions would allow it to be used as a "normal LLM". Further fine-tuning on other tasks and outputs can increase its range of use and general capabilities, making it more "worth" to use.

Lastly, the experience gain from this research is the main takeaway, and we are really satisfied with the results we achieved. We believe that the knowledge we acquired is just an headstart for the bigger picture we have in mind.

### Resources
Links to the repositories we used:
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [LLAMA.cpp](https://github.com/ggerganov/llama.cpp)

Links to our [HuggingFace](https://huggingface.co/SimplyLeo/Zephyr-Fraudulence-Detector) models:
- [Zephyr-Fraudulence-Detector-F16](https://huggingface.co/SimplyLeo/Zephyr-Fraudulence-Detector/blob/main/zehpyr-fraudulence-detector-f16.gguf)
- [Zephyr-Fraudulence-Detector-Q8](https://huggingface.co/SimplyLeo/Zephyr-Fraudulence-Detector/blob/main/zehpyr-fraudulence-detector-q8_0.gguf)