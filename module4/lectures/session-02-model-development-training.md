# Session 2 — Model Development and Training

**AI in Healthcare · Unit IV · B.Tech AI, Third Year**  
**Student lecture notes and primary study material**  
**Previous session:** [Session 1 — The healthcare AI lifecycle and data requirements](session-01-lifecycle-data-requirements.md)  
**Companion practical:** [Lab 12 — Training an imaging model: from scratch versus transfer learning](../labs/lab-12-imaging-model-training.ipynb)

## Learning objectives

After studying this chapter, you should be able to:

1. Describe model training as a sequence of decisions, and state which data each decision may use.
2. Choose a sensible starting model and baseline for tabular, signal, image, and text tasks.
3. Explain how class imbalance, overfitting, and model selection are handled without using the test set.
4. Explain why imaging models are usually built by transfer learning, and how augmentation can help or harm.
5. Compare prompting, few-shot prompting, and fine-tuning (including LoRA) for adapting a foundation model, and explain why every adaptation must be re-validated.

You need Session 1 of this unit, Lab 4 (the heart-disease referral model), the Unit II ECG beat-classification lecture and Lab 6, and the Unit III imaging session (MedGemma). Medical knowledge is not assumed.

## Contents

1. [From data to a frozen model](#1-from-data-to-a-frozen-model)
2. [Training is a sequence of decisions](#2-training-is-a-sequence-of-decisions)
3. [Split first, and split by the right unit](#3-split-first-and-split-by-the-right-unit)
4. [Choosing a model family and a baseline](#4-choosing-a-model-family-and-a-baseline)
5. [Training choices that matter in healthcare](#5-training-choices-that-matter-in-healthcare)
6. [Training models on medical images](#6-training-models-on-medical-images)
7. [Adapting foundation models](#7-adapting-foundation-models)
8. [Revision summary](#8-revision-summary)
9. [Review questions](#9-review-questions)
10. [Further reading](#10-further-reading)

## 1. From data to a frozen model

Session 1 ended with a dataset whose population, labels, and quality had been checked and documented. This session covers the next lifecycle stage: turning that data into a model that is **frozen** — fixed, with no further changes — before it is tested and validated.

You have already trained three kinds of model in this course:

| Model | Course material | Input | Output |
|---|---|---|---|
| Logistic regression | Lab 4 | 13 clinical measurements of one patient | Probability of coronary heart disease |
| 1D convolutional neural network (CNN) | Unit II ECG lecture, Lab 6 | One heartbeat, 187 signal samples | One of five beat classes |
| Medical foundation model (MedGemma) | Unit III, session 2 | An image and a text prompt | Generated text |

The mechanics were different each time, but the same questions came up: how to split the data, which model to try first, how to deal with rare classes, and when to stop training. This session makes those questions explicit, then applies them to a new case — training an image classifier — which you will do yourself in Lab 12.

## 2. Training is a sequence of decisions

### 2.1 The pipeline

Every supervised model in this course followed the same pipeline:

```text
  Checked dataset (Session 1)
          |
          v
  1. Split into training / validation / test sets
          |
          v
  2. Fit preprocessing on the training set only
          |
          v
  3. Choose a model family and a baseline
          |
          v
  4. Train on the training set
          |
          v
  5. Compare settings and models on the validation set
          |
          v
  6. Freeze the chosen model, preprocessing, and threshold
          |
          v
  7. Evaluate once on the test set   --> Session 3: validation
```

The three sets have different jobs:

- The **training set** is used to fit the model's parameters.
- The **validation set** is used to choose between models, settings, and thresholds.
- The **test set** is used once, at the end, to estimate performance on new data.

### 2.2 Each step is a decision that must be recorded

Each step involves a choice. A choice that is not recorded cannot be checked or repeated.

| Decision | Example | What goes wrong if it is done badly | Record |
|---|---|---|---|
| How to split | 70/15/15 by patient | The same patient appears in training and test data | Unit of splitting, random seed |
| Preprocessing | Mean imputation, scaling | Test-set statistics leak into training | What was fitted, and on which set |
| Model family | Logistic regression | A complex model is used without beating a simple one | Baseline and its result |
| Imbalance handling | Class weights | The model ignores the rare class | Weighting method |
| When to stop | Best validation epoch | The model memorises the training data | Number of epochs, stopping rule |
| Settings | Learning rate, layer sizes | Settings are tuned on the test set | Search range, chosen values |
| Threshold | Probability ≥ 0.3 means "refer" | The threshold ignores the clinical trade-off | Threshold and why it was chosen |

The last column becomes part of the **model card** introduced in Session 1.

## 3. Split first, and split by the right unit

### 3.1 Why splitting comes first

Any step that looks at the data — imputing missing values, scaling, choosing features — can leak information from the test set into the model if it happens before the split. In Lab 4 you fitted `StandardScaler` on the training data only, then applied it to the test data. That rule applies to every learned preprocessing step.

### 3.2 Split by patient, not by row

A model will be used on **new patients**. The test set should therefore contain patients the model has never seen.

When one patient contributes many rows, a random row-level split puts the same patient's data on both sides. The model can then recognise that patient rather than the condition, and test performance looks better than it will be in practice.

| Dataset | One row is | Can we split by patient? |
|---|---|---|
| UCI heart disease (Lab 4) | One patient | Rows are patients, but the files have no identifier to check duplicates (Session 1) |
| MIT-BIH beats (Unit II, Lab 6) | One heartbeat | No: the original 48 recordings came from 47 people, but the CSV files contain no recording identifier |
| PneumoniaMNIST (Lab 12) | One chest X-ray | No: the published files contain no patient identifier |

In all three course datasets, a proper patient-level split is not possible with the files we have. This is a common situation with public datasets. The correct response is to **state the limitation**, not to ignore it. Session 3 returns to this when we discuss internal and external validation.

## 4. Choosing a model family and a baseline

### 4.1 Start from the data type

| Data type | Typical starting model | Why |
|---|---|---|
| Tabular (one row of measurements) | Logistic regression, gradient-boosted trees | Few features; strong, interpretable baselines |
| Signals (ECG, EEG) | 1D CNN | Learns local waveform shapes, such as a wide QRS complex |
| Images (X-ray, MRI slice) | 2D CNN, usually pretrained | Learns local visual patterns; pretraining reduces the data needed |
| Text, or images with open-ended questions | Foundation model | Already trained on large data; adapted rather than trained from scratch |

### 4.2 Always train a baseline

A **baseline** is a simple model that a more complex model must beat. Two kinds are useful:

- A **trivial baseline**, such as always predicting the most common class. If 74% of training images show pneumonia, a "model" that always says pneumonia is 74% accurate and completely useless.
- A **simple learned baseline**, such as logistic regression. In Unit II, a classical model was compared with neural networks on the same ECG split.

If a deep network does not clearly beat the simple baseline on validation data, the extra complexity — training cost, harder explanation, more ways to fail — is not justified.

## 5. Training choices that matter in healthcare

### 5.1 Class imbalance

Clinical outcomes are often rare. A model trained with an ordinary loss function can reach high accuracy by mostly predicting the common class.

Two common responses:

- **Class weights** make errors on the rare class cost more in the loss function. The Unit II ECG notebook compared several weighting schemes.
- **Resampling** changes how often each class appears during training, for example by repeating rare-class examples.

Either changes the model's predicted probabilities. A weighted model's probabilities no longer match the true outcome rate, which matters for calibration (Session 3). Resampling must be applied **only to the training set**; the validation and test sets keep their real class proportions.

### 5.2 Overfitting and when to stop

A model **overfits** when it fits noise in the training data and performs worse on new data. Session 1 showed why this is a particular risk in healthcare: few outcome events relative to the number of parameters.

For neural networks, the usual signal is in the **learning curves** — training and validation loss plotted against epochs. One **epoch** is one pass through the training set.

```text
 loss
  |  \
  |   \ .  validation loss
  |    \  .            .  .  .
  |     \    .  .  .
  |      \                       <-- validation loss rises: overfitting
  |       \__
  |          \______  training loss keeps falling
  +----------------------------------------> epoch
               ^
         best validation epoch
```

Common responses:

- **Early stopping:** keep the model from the epoch with the best validation result.
- **Regularisation:** penalise large weights (weight decay) or randomly switch off units during training (dropout).
- **More or more varied data:** including augmentation for images (Section 6.4).

### 5.3 Choose settings on validation data only

Settings that are not learned from the data — learning rate, number of layers, class-weight scheme, threshold — are called **hyperparameters**. Trying many settings and picking the best one is a search, and the search itself can overfit. If the search uses the test set, the test result is no longer an honest estimate.

The rule is: **compare on validation data, freeze, then test once.** If you look at the test result and then change the model, the test set has become a second validation set, and you need a new test set.

### 5.4 Reproducibility

Another person should be able to rerun training and obtain the same model, or very nearly. Record:

- random seeds for the split, weight initialisation, and data order;
- library versions and hardware (GPU training can differ slightly between runs even with fixed seeds);
- every hyperparameter and the stopping rule;
- the saved model weights and preprocessing objects, with the data version used.

## 6. Training models on medical images

### 6.1 A concrete task

**Pneumonia** is an infection that inflames the lungs. On a chest X-ray, infected regions often look whiter (more opaque) than healthy lung. Lab 12 uses **PneumoniaMNIST**, a public dataset of paediatric chest X-rays, each labelled *normal* or *pneumonia*.

| Element | Lab 12 |
|---|---|
| Computational task | Binary image classification |
| Unit of analysis | One chest X-ray image |
| Images | 4,708 training, 524 validation, 624 test |
| Resolution used | 64 × 64 pixels, greyscale |
| Source | 5,856 paediatric chest X-rays (Kermany et al., 2018), resized by the MedMNIST project |

This is an educational task. A 64 × 64 image is far below the resolution a radiologist reads, and the model is not a diagnostic tool.

### 6.2 Why images are harder than tables

- **Many inputs.** A 64 × 64 image has 4,096 pixel values; a clinical X-ray has millions. The heart-disease model had 13.
- **Expensive labels.** Each label needs an expert to read the image (recall the annotation discussion in Session 1).
- **Compute.** Training a CNN on thousands of images takes minutes on a GPU (graphics processing unit) and much longer on a CPU.

The central problem is that a CNN has **many parameters** relative to the number of labelled images. A standard ResNet-18 network has about 11.7 million parameters. Lab 12 has 4,708 training images.

### 6.3 Transfer learning

**Transfer learning** reuses a network that was already trained on a large dataset for a different task. The standard source is **ImageNet**, a collection of over a million labelled photographs of everyday objects.

The early layers of such a network detect general patterns — edges, textures, simple shapes — that are also useful in X-rays. Only the later layers are specific to the original task.

There are two common ways to reuse the network:

| Approach | What is trained | When it suits |
|---|---|---|
| **Feature extraction** | Only a new final layer; the pretrained layers are frozen | Very little data; fast |
| **Fine-tuning** | All layers, usually with a small learning rate | More data; images differ a lot from the source domain |

A small calculation shows why feature extraction is cheap. ResNet-18 ends with 512 features. A new final layer for one output (pneumonia probability) has 512 weights plus 1 bias:

| Quantity | Value |
|---|---|
| Parameters trained in feature extraction | 512 + 1 = 513 |
| Parameters trained in full fine-tuning | about 11.7 million |

Medical images differ from photographs: they are often greyscale, and the meaningful differences can be subtle. Transfer learning from ImageNet is a common, practical starting point, but it is not guaranteed to help; Raghu et al. (2019) found that its benefit varies with the task and the amount of data. In Lab 12 you will measure the difference yourself.

### 6.4 Augmentation must keep the label true

**Data augmentation** creates modified copies of training images — rotated, shifted, slightly brighter — so the model sees more variety. It is applied only during training.

An augmentation is acceptable only if the modified image is still a realistic example **with the same label**:

| Augmentation | Chest X-ray | Reason |
|---|---|---|
| Small rotation (a few degrees) | Usually acceptable | Patients are not always positioned perfectly straight |
| Small shift or crop | Usually acceptable | If the crop does not cut off the lung region |
| Small brightness or contrast change | Usually acceptable | Exposure varies between machines |
| Left–right flip | Questionable | The heart appears on the wrong side, producing an anatomy rarely seen in real patients |
| Large rotation (90°) | Not acceptable | No real chest X-ray looks like this |

The right choice depends on the task. A flip is harmless for a skin-lesion photograph but changes anatomy in a chest X-ray.

### 6.5 Limitations of the Lab 12 data

Keep three facts in mind when you interpret your Lab 12 results:

1. **No patient identifiers.** MedMNIST created its validation set by randomly splitting the source training set 9:1. Images from the same child may appear in both the training and validation sets, which can make validation results optimistic.
2. **The test set is a separate source set.** MedMNIST uses the source dataset's separate held-out set as its test set. It is still from the same hospital population, so it is not an external validation.
3. **The class balance is not a clinical prevalence.** The proportion of pneumonia images reflects how the dataset was assembled, not how often children seen in a clinic have pneumonia.

## 7. Adapting foundation models

### 7.1 The adaptation ladder

In Unit II, Session 5, you compared prompting, retrieval, and fine-tuning for medical foundation models, and in Unit III you evaluated MedGemma on images without changing it. Here we look at the steps that involve more training.

Move up the ladder only when a lower step has been tried, evaluated, and found insufficient:

| Step | What changes | Data needed | Compute | Main risk |
|---|---|---|---|---|
| Zero-shot prompting | The instructions only | A validation set to evaluate | Inference only | Ignores instructions; invents facts |
| Few-shot prompting | Examples added to the prompt | A few reviewed examples, taken from development data | Inference only | Examples bias the output; longer prompts |
| Fine-tuning with an adapter (e.g. LoRA) | A small number of added weights | Hundreds to thousands of reviewed examples | One GPU, often | Overfits small data; forgets general abilities |
| Full fine-tuning | All weights | More data than adapter tuning | Several GPUs, usually | Same, with higher cost |

### 7.2 What LoRA does

**LoRA** (Low-Rank Adaptation; Hu et al., 2021) keeps the original weights frozen. Next to a weight matrix it adds two small matrices whose product is a correction to that matrix. Only the small matrices are trained.

For one weight matrix of size 4,096 × 4,096 and a LoRA **rank** of 8 (the inner size of the two small matrices):

| Quantity | Calculation | Parameters |
|---|---|---|
| Original matrix (frozen) | 4,096 × 4,096 | 16,777,216 |
| LoRA matrices (trained) | 4,096 × 8 + 8 × 4,096 | 65,536 |
| Fraction trained | 65,536 ÷ 16,777,216 | about 0.4% |

This is why adapter fine-tuning fits on a single GPU, and why one base model can carry several small task-specific adapters. The trade-off is that a low rank limits how much the model can change.

### 7.3 Every adaptation must be re-validated

Fine-tuning produces a **new model**. Results obtained for the original model — benchmark scores, error analyses, safety checks — do not carry over automatically. After any adaptation:

- compare against the unadapted model and a simple baseline on the same validation data;
- check for **forgetting**: abilities the base model had that the tuned model lost;
- keep a test set that was not used for choosing prompts, examples, or tuning settings;
- update the model card with the adaptation data, method, and settings.

## 8. Revision summary

| Concept | Key idea |
|---|---|
| Three data sets | Train to fit, validate to choose, test once to estimate |
| Splitting unit | Split by patient when patients have several rows; state it when you cannot |
| Baseline | A trivial and a simple model set the bar a complex model must clear |
| Imbalance | Class weights or resampling, on the training set only |
| Overfitting | Watch validation loss; stop early; regularise |
| Hyperparameters | Chosen on validation data; test set never used for choices |
| Transfer learning | Reuse a pretrained network; train the head, or fine-tune everything |
| Augmentation | Only transformations that keep the image realistic and the label true |
| LoRA | Train small added matrices; base weights frozen |
| Re-validation | An adapted model is a new model |

## 9. Review questions

1. A team scales its features using the mean and standard deviation of the whole dataset, then splits it. What has gone wrong, and how would it affect the test result?
2. A dataset contains 20 X-rays per patient from 500 patients. Describe a split that gives an honest estimate for new patients.
3. 74% of training images show pneumonia. What accuracy does the trivial baseline achieve, and why does that number make accuracy a poor main metric here?
4. Validation loss is lowest at epoch 6 and rises afterwards, while training loss keeps falling. Which epoch's model should be kept, and why?
5. A student tries 30 learning rates and reports the best test-set result. Why is that result optimistic?
6. Explain the difference between feature extraction and fine-tuning. How many parameters are trained when a 512-feature network gets a new one-output final layer?
7. Is a left–right flip a suitable augmentation for chest X-rays? For skin-lesion photographs? Explain.
8. A LoRA adapter of rank 4 is added to a 2,048 × 2,048 weight matrix. How many parameters does it train?
9. A fine-tuned MedGemma variant scores higher than the base model on the team's validation set. Give two further checks before the team claims it is better.

## 10. Further reading

- Kermany DS et al. [Identifying medical diagnoses and treatable diseases by image-based deep learning](https://doi.org/10.1016/j.cell.2018.02.010). *Cell* 2018;172(5):1122–1131.e9. — the source of the paediatric chest X-rays used in Lab 12.
- Yang J et al. [MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification](https://doi.org/10.1038/s41597-022-01721-8). *Scientific Data* 2023;10:41. — how PneumoniaMNIST was built and split; see also [medmnist.com](https://medmnist.com/).
- Yosinski J et al. [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792) *NeurIPS* 2014. — what early and late layers learn.
- Raghu M et al. [Transfusion: Understanding transfer learning for medical imaging](https://arxiv.org/abs/1902.07208). *NeurIPS* 2019. — when ImageNet pretraining does and does not help.
- Hu EJ et al. [LoRA: Low-rank adaptation of large language models](https://arxiv.org/abs/2106.09685). 2021.
- Riley RD et al. [Calculating the sample size required for developing a clinical prediction model](https://www.bmj.com/content/368/bmj.m441). *BMJ* 2020;368:m441. — links overfitting to the number of outcome events.
- [Unit II, Session 5 — Medical foundation models](../../module2/lectures/session-05-medical-foundation-models.ipynb), Section 3, for prompting, retrieval, and fine-tuning compared.
