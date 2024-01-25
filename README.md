# CFFitST: Classification Few-Shot Fit Sentence Transformer

This repository houses the code implementation for CFFitST, a novel approach for few-shot text classification using Sentence Transformers. Leveraging the power of Sentence Transformers, CFFitST can be initialized with either a pre-trained Sentence Transformer or a base embedding model such as RoBERTa. This solution has been tailored for the few-shot issue report classification tool competition in the NLBSE 2024 competition held in Lisbon.

## Primary characteristics

- Sample Generation: CFFitST dinamically generates pairs of examples to refine a Sentence Transformer. This iterative approach utilizes the performance feedback from the previous iteration, employing a minimum cosine similarity threshold as a criterion.

- Sentence Embeddings: CFFitST leverages Sentence Transformers to generate robust sentence embeddings. Choose between a pre-trained Sentence Transformer or a base embedding model like RoBERTa.

- Classification Head: The baseline CFFitST implementation includes a Classification Head, employing logistic regression with softmax as the activation function.

## Inspirations

Inspired by the remarkable success of SetFit in the NLBSE 2023 competition. Our model, designed to emulate SetFit's strengths, integrates an adaptive sampling mechanism, contributing to its enhanced performance.

## Few-Shot Classification

CFFitST is purpose-built to tackle few-shot classification problems, specifically those with a larger number of training sentences per class (greater than 50 sentences). In contrast to SetFit, which excels with smaller data fractions, CFFitST demonstrates improved performance as the size of the training data increases. Our primary objective is to enhance the F1-Score for scenarios where training data is not severely limited but still poses a few-shot classification challenge.

## Usage

### Dependencies Installation

Ensure you are in the project directory where the ``requirements.txt`` file is located.

```bash
pip install -r requirements.txt
```
### Run template notebook
To execute the template notebook for the NLBSE issue report classification competition, open your notebook application and load the file named ``CFFitST_Template.ipynb``. Run all the cells in the notebook to execute the code. The template is specifically designed for the NLBSE issue report classification competition.

Navigate to the second cell in the notebook and locate the following line of code:

```python
OUTPUT_PATH = 'output' # output directory
```

Change the value of ``OUTPUT_PATH`` to specify the desired output directory where the sentence transformers models and the results file will be saved. Once you have configured the output directory, execute the modified cell.

Check the specified output directory for the saved sentence transformers models. Additionally, a file named results.json containing the results will be available in the same directory.

This notebook is specifically designed for the NLBSE 2024 issue report classification tool competition. It includes the directory containing the relevant data for this competition, along with the recommended template structure.