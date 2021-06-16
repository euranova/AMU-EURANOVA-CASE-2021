# Event Extraction

This repository holds a Python module whose goal is to perform event extraction on different dataset.

It has two different classes in function of the problem category, sequence classification (such as document classification) and token classification with BIO format.
We are using Transformers library for models, so models from the [HuggingFace model hub](https://huggingface.co/models) are available.

It can be used on the [ProtestNews2019](https://emw.ku.edu.tr/clef-protestnews-2019/) dataset or the [task 1 of CASE 2021 (subtask 1, 2 and 4.)](https://github.com/emerging-welfare/case-2021-shared-task), to obtain them you need to ask the task organizers as we can't share them.

## Usage

### Docker

For an easy use a Dockerfile is provided.

You can install it with the following commands, the interface of MLflow will be available on the port 5000:

```bash
docker build -t event_extraction .
docker run -it --gpus all -v local_path_to_repo:/event_extraction/ --shm-size=50gb --name case_2021 event_extraction /bin/bash
```

If you want to change the port of MLflow you can add the command `-p your_port:5000`.

You can then access your docker with the command:
```bash
docker attach case_2021
```

If you want to quit the container without closing it, the shortcut is `ctrl+P ctrl+Q`.

### Library

The library can also be installed directly to be used. 

Depending on your usage there are 2 versions of the requirements, one for GPU the other for CPU only.

You can install them with this command, you just have to choose the environment you want:

```bash
conda env create -f {environment.yml|environment_cpu.yml}
```

And then you can activate the environment and then install the library with the commands:

```bash
conda activate protestnews
pip install -e .
```

You will then be able to run the scripts or to use the library in your own code.

## Content

### Data Loader

The data_loader part of the library contains the code used to load a dataset. For now, the datasets from ProtestNews 2019 and from CASE 2021 are available. Some general utility functions are also available.

You can also create another class which outputs the same type of data, i.e. a dictionary with the different set available in it. Or you can directly feed the data in the correct format into the models.

### Model

The model part of the library contains the code to train, evaluate and predict a specific model.

There are two different files depending on the type of task. For document or sentence classification you can use *bert_sequence_classif*,
for named entity recognition you can use *bert_token_classif*. These models works with Pytorch-based Transformers library.

The *bert_sequence_classif* will take as input:
 - the name of the Transformers model you want to use. Many models are available on the [HuggingFace model hub](https://huggingface.co/models). (warning: some models requires some specific options later on which are not activated here, such as RoBERTa. All the BERT models will work.)
 - a TrainingArguments object from Transformers library, information concerning this object can be found in [the documentation of Transformers library](https://huggingface.co/transformers/main_classes/trainer.html?highlight=trainingarguments#transformers.TrainingArguments).

The *bert_token_classif* will take as input:
 - the name of the Transformers model you want to use. Many models are available on the [HuggingFace model hub](https://huggingface.co/models). (warning: some models requires some specific options later on which are not activated here, such as RoBERTa. All the BERT models will work.)
 - a TrainingArguments object from Transformers library, information concerning this object can be found in [the documentation of Transformers library](https://huggingface.co/transformers/main_classes/trainer.html?highlight=trainingarguments#transformers.TrainingArguments).
 - a string representing the loss. *macro* for soft macro F1 loss, *micro* for soft micro F1 loss (which does not seem to work) and *base* for the base loss of the transformers library.
 - an integer representing the general seed of the model. This seed is constant during our experimentations about the stability in the paper.
 - an integer representing the seed controling the initialization of the prediction layer.

```python
from transformers import TrainingArguments

from event_detection.bert_token_classif import BertTokenClassif

training_arguments = TrainingArguments(output_dir="results")

classifier = BertTokenClassif(pretrained_model="distilbert-base-uncased", training_arguments=training_arguments)
```

Then, depending on your model, you can train them or use use them to predict on same data. 

For *bert_sequence_classif*, the texts are supposed to be a list with one document in each element as an unique string, the labels a list of 0 and 1.

For *bert_token_classif*, the texts are supposed to be a list of list with each word in an unique string, the labels a list of list with the corresponding labels in string.

To train/fine-tune a model:

```python
classifier.train(train_text, train_label, eval_text, eval_label)
```

To test a model:

```python
classifier.test(test_text, test_label)
```

### Script

The script folder contains some script to launch the training and evaluation of the first and third task of ProtestNews2019. To launch these scripts we use MLflow with MLproject file.

To launch a script:

```bash
mlflow run -e entry_point_name . --experiment-name exp_name -P parameter_name=value -P parameter2_name=value
```

#### Behavioural fine-tuning for sub-task 4

Some of the scripts can be used to do something close to behavioural fine-tuning. In our case we do not keep the classification layer.

To do this, we need to do two different learning phase, the first one is with the *2021_task4_behavioural* entrypoint:

```bash
mlflow run -e 2021_task4_behavioural . --experiment_name behavioural_phase1 -P epochs=1
```

Then, you need to load the model learned with this command with the *2021_task4* entrypoint. You can find where the model was saved with the MLflow interface, and the "output_dir" value of the experiment:


```bash
mlflow run -e 2021_task4 . --experiment_name behavioural_phase2 -P epochs=20 -P model_name=results/2021_task_4_behavioural/2021-04-30_09-48-02 --dataset=train_only
```

## Citation

If you use our work please cite:

```bibtex
@inproceedings{BouscarratAssessing, 
    title = "AMU-EURANOVA at CASE 2021 Task 1: Assessing the stability of multilingual BERT",
    author = {Bouscarrat, L{\'e}o and Bonnefoy, Antoine and Capponi, C{\'e}cile and Ramisch, Carlos},
    booktitle = "Proceedings of the 4th Workshop on Challenges and Applications of Automated Extraction of Socio-political Events from Text (CASE 2021)",
    month = aug,
    year = "2021",
    address = "online",
    publisher = "Association for Computational Linguistics (ACL)",
}
```

You can also find all the [citations related to the datasets we are using here](https://github.com/emerging-welfare/case-2021-shared-task/blob/main/task1/publications.bib).