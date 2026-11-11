# Session 1 — The Healthcare AI Lifecycle and Data Requirements

**AI in Healthcare · Unit IV · B.Tech AI, Third Year**  
**Student lecture notes and primary study material**  
**Running example:** [Lab 4 — Clinical Decision Support: heart-disease referral alerting](../../module2/labs/heart_disease_cds_lab.ipynb)

## Learning objectives

After studying this chapter, you should be able to:

1. Describe the stages of the healthcare AI lifecycle and explain why it is a loop rather than a straight line.
2. State the data requirements for a defined prediction task: population, decision-time availability, and number of outcome events.
3. Explain where labels come from and why a label can be a weak proxy for the condition of interest.
4. Classify data-quality problems as conformance, completeness, or plausibility problems.
5. Explain what a datasheet and a model card record, and why they matter later in the lifecycle.

You need Unit I (healthcare data sources, missing values) and Lab 4 (the heart-disease referral model). Medical knowledge is not assumed.

## Contents

1. [The running example](#1-the-running-example)
2. [The healthcare AI lifecycle](#2-the-healthcare-ai-lifecycle)
3. [Data collection requirements](#3-data-collection-requirements)
4. [Labels and reference standards](#4-labels-and-reference-standards)
5. [Data quality requirements](#5-data-quality-requirements)
6. [Documentation as a lifecycle output](#6-documentation-as-a-lifecycle-output)
7. [Revision summary](#7-revision-summary)
8. [Review questions](#8-review-questions)
9. [Further reading](#9-further-reading)

## 1. The running example

In Lab 4 you built a model for a general physician's clinic. The physician sees adults for routine checkups and must decide whether to **refer** a patient to a cardiologist for further, invasive evaluation. The model flags patients whose predicted risk of coronary heart disease is high.

You trained it on the UCI Heart Disease (Cleveland) dataset: 303 patients, 13 clinical attributes, and a disease label. The lab focused on cleaning, encoding, training, and choosing an alert threshold.

This session asks an earlier question: **before any training, what data does this project need, and how would we know the data is good enough?** We will return to the same dataset throughout.

## 2. The healthcare AI lifecycle

### 2.1 The stages

A healthcare AI system passes through a sequence of stages. Each stage produces something the next stage depends on.

```text
  1. Clinical need and problem framing
              |
              v
  2. Data collection and quality assessment   <-- this session
              |
              v
  3. Model development and training
              |
              v
  4. Validation
              |
              v
  5. Deployment and workflow integration
              |
              v
  6. Monitoring and updating  ---------------+
              |                              |
              v                              |
  7. Retirement                              |
                                             |
  Problems found in monitoring send the team back to
  stage 1, 2, or 3  <------------------------+
```

The loop at the bottom matters. A deployed model can become worse when patients, equipment, coding practices, or clinical guidelines change. Monitoring detects this, and the team returns to an earlier stage: new data, retraining, or even a reframed problem. Eventually a model may be **retired** because it is no longer safe or useful.

### 2.2 What makes the healthcare lifecycle different?

The stages resemble any machine-learning project, but three features are specific to healthcare:

- **Clinical involvement at every stage.** Clinicians define the need, judge whether labels are meaningful, and decide whether the output can be acted on.
- **An evidence trail.** Each decision — which patients, which label, which threshold — must be documented, because regulators, hospitals, and patients may later ask why the system behaves as it does.
- **Change after deployment.** Updating a model that affects patient care is a controlled process, not a silent software update.

Regulators summarise these expectations as *Good Machine Learning Practice* (GMLP). The guiding principles were published by the US FDA, Health Canada, and the UK MHRA in 2021 and finalised internationally by the International Medical Device Regulators Forum (IMDRF) in January 2025. Unit V covers regulation in detail.

### 2.3 Where each stage is taught

| Stage | Where in the course |
|---|---|
| Problem framing | Unit I, session 7; recapped below |
| Data collection and quality | **This session** |
| Development and training | Unit IV, session 2; Lab 12 |
| Validation and data challenges | Unit IV, session 3; Lab 13 |
| Deployment and workflow integration | Unit IV, session 4 |
| Explainability | Unit IV, session 5; Lab 14 |
| Monitoring, regulation, governance | Unit V |

### 2.4 Problem framing decides the data

You met task definition in Unit I. Here is the framing for the running example:

| Element | Heart-disease referral model |
|---|---|
| Intended user | General physician |
| Setting | Routine outpatient checkup |
| Unit of analysis | One patient at one checkup |
| Decision time | End of the checkup, when referral is decided |
| Target | Coronary heart disease present (yes/no) |
| Action | Refer to a cardiologist, or not |

Every data requirement in the next section follows from this table. If the setting or decision time changes, the data requirements change too.

## 3. Data collection requirements

### 3.1 The data must represent the intended population

A model learns the patterns of the patients in its training data. If those patients differ from the patients it will be used on, its predictions may not transfer.

Compare the framing with the Cleveland data:

- **Who was included?** Every patient in the dataset has an angiography result — an invasive X-ray test of the heart's arteries. So these were patients who had already been sent for angiography, not a general checkup population.
- **Sex balance.** 206 of the 303 patients (68%) are male.
- **Place and time.** One clinic, with data donated to the UCI repository in 1988.

The UCI collection also includes data from three other sites. The proportion of patients with disease differs widely between them:

| Site | Patients | With disease |
|---|---|---|
| Cleveland Clinic Foundation, USA | 303 | 46% |
| Hungarian Institute of Cardiology, Budapest | 294 | 36% |
| V.A. Medical Center, Long Beach, USA | 200 | 75% |
| University Hospital, Zurich, Switzerland | 123 | 94% |

A routine checkup clinic would see far fewer patients with disease than any of these sites. A model trained here will have learned risk levels that are probably too high for that setting. *(Percentages computed from the UCI processed files.)*

**Requirement:** state the intended population, then check who the data actually contains — by site, sex, age, and the route by which patients entered the dataset.

### 3.2 The inputs must be available at decision time

A model can only use information that exists at the moment the decision is made. You saw this in Unit I: a lab result belongs to the time it becomes *available*, not the time the sample was taken.

Two Cleveland columns need checking:

- `ca` is the number of major arteries seen on **fluoroscopy** (live X-ray imaging). The documentation does not say when it was recorded. If it came from the angiography procedure, it is part of the very test the referral decides on — and would not exist at the checkup.
- `thal` is the result of a **thallium stress test**, a nuclear-imaging test. It is not normally part of a routine checkup.

Neither column is necessarily wrong to collect. The point is that a model trained with them assumes they are available. If they are not, the model cannot be used as designed, or it quietly learns from information it will never see in practice.

**Requirement:** for every input, record when and where it is measured and when the result reaches the user. Remove or replace inputs that are not available at decision time.

### 3.3 There must be enough outcome events

For a classification model, the number of patients **with the outcome** matters more than the total number of rows. These patients are called **events**. In the Cleveland data, an event is a patient with heart disease: 139 of the 303 patients.

**Why events, not rows?** To predict disease, the model must learn how patients with disease differ from patients without it. The information for that comes mostly from the smaller of the two groups. If only 5 of 10,000 patients had the outcome, the model would know almost nothing about what the outcome looks like, however many rows there were. (Strictly, the limit is the smaller class. Usually, as here, that is the patients with the outcome.)

**What is a parameter?** A **parameter** is a number the model estimates from the data. In logistic regression, each input column the model sees gets one coefficient. This is not the same as the number of original attributes, because encoding adds columns. Chest-pain type (`cp`) has 4 categories. One-hot encoding with `drop_first=True` turns it into 3 separate 0/1 columns, so it costs 3 parameters. That is how Lab 4's 13 attributes become 18 parameters:

| Attributes | Columns after encoding |
|---|---|
| 9 numeric or binary attributes | 9 |
| `cp` (4 categories) | 3 |
| `restecg`, `slope`, `thal` (3 categories each) | 2 + 2 + 2 = 6 |
| **Total parameters** (the intercept is not counted) | **18** |

**Events per parameter (EPP)** divides the events available for training by the parameters to estimate:

| Quantity | Value |
|---|---|
| Patients in Cleveland data | 303 |
| Patients with disease (events) | 139 |
| Events in an 80% training split | 0.8 × 139 ≈ 111 |
| Predictor parameters after Lab 4's one-hot encoding | 18 |
| **Events per parameter** | **111 ÷ 18 ≈ 6** |

On average, each coefficient is estimated from the information in about six patients with disease.

**Why a low EPP is a problem.** With few events per parameter, the model has enough freedom to fit **noise**: quirks of these particular patients that will not recur in new ones. This is **overfitting**, and it shows up in three ways:

- Performance measured on the training data looks better than it really is (**optimism**).
- Coefficients are unstable. A slightly different training sample could change them a lot, or even flip their sign.
- Predicted risks are too extreme: high-risk patients get scores that are too high and low-risk patients get scores that are too low. This is poor **calibration**, covered in session 3.

**How many events are enough?** An older rule of thumb asked for at least 10 events per parameter. Lab 4 would then need about 18 × 10 = 180 training events instead of 111. Current guidance (Riley et al., 2020) treats a single fixed number as too crude. It calculates the minimum sample size from the outcome rate, the number of parameters, and the performance the model is expected to reach. The aim is to keep overfitting small and to estimate the overall risk level precisely. The result can be above or below what the 10-EPP rule gives.

**The rare-outcome trap.** A dataset that sounds large can still be small. A dataset of 2,000 patients with an outcome rate of 5% contains only 2,000 × 0.05 = 100 events. With the same 18 parameters, that is 100 ÷ 18 ≈ 5.6 events per parameter — no better than Cleveland, despite more than six times as many patients. Many clinical outcomes, such as sepsis, readmission, and many cancers, are rare.

**When events are scarce**, the options are:

1. Use fewer parameters: choose predictors for clinical reasons before modelling, or merge rare categories.
2. Use a penalised model (ridge or lasso), which shrinks coefficients and reduces overfitting. This helps but does not replace adequate data.
3. Collect more data, with the sample size planned in advance.

**Requirement:** count events, not rows, and plan the sample size before collecting data.

### 3.4 Access and permission

Data collection also needs legal and ethical permission: consent or an approved exemption, de-identification, and an agreement on how the data may be used. These requirements are covered in Unit V (privacy and data governance, including India's ABDM and DPDP rules).

## 4. Labels and reference standards

### 4.1 Where labels come from

A **label** is the value the model learns to predict. Healthcare labels are produced in different ways, each with a cost and a weakness:

| Label source | Example | Weakness |
|---|---|---|
| Diagnosis or billing codes | ICD code for coronary disease | Recorded for payment; may be missing or inaccurate |
| Chart review | A clinician reads each record | Slow; reviewers may disagree |
| Expert annotation | Radiologists mark images (Unit III) | Expensive; requires agreement checks |
| Reference-standard test | Angiography result | Invasive, so only some patients receive it |
| Follow-up outcome | Heart attack within 5 years | Requires long follow-up; patients are lost |

A **reference standard** is the best available method for deciding whether a condition is present. In the Cleveland data, the label is **angiographic disease status**: disease is present if any major artery is narrowed by more than 50% of its diameter.

### 4.2 Proxy labels

A **proxy label** stands in for what we actually care about. For example, "was billed for pneumonia" is a proxy for "had pneumonia".

Even the Cleveland label is a simplification. The original value ranges from 0 to 4; Lab 4 merged values 1–4 into "disease present". The 50% threshold is itself a convention. A patient just under the threshold is labelled "no disease", even though their arteries may be narrowed.

When labels come from people — chart reviewers or radiologists — the team should measure how often independent reviewers agree. Low agreement means the model is learning from inconsistent labels.

**Requirement:** state exactly how the label is defined, who or what produced it, and which patients could receive it.

## 5. Data quality requirements

### 5.1 Three categories of data-quality problems

Kahn et al. (2016) proposed a shared vocabulary for checking healthcare data. It groups problems into three categories:

- **Conformance** — do values follow the expected format, allowed values, and structure?
- **Completeness** — are the values present?
- **Plausibility** — are the values believable?

Each can be checked by **verification**, using rules from inside the dataset (allowed codes, data types), or **validation**, using an external reference (published population statistics, another trusted source).

### 5.2 Examples from the heart-disease data

The four UCI sites show every category. *(Counts computed from the UCI processed files.)*

| Category | Check | What we find |
|---|---|---|
| Value conformance | `cp` must be 1, 2, 3, or 4 | Missing values are written as `?`, not as a number or blank |
| Relational conformance | Each patient appears once | The files contain **no patient identifier**, so duplicates cannot be checked |
| Computational conformance | Derived values are calculated correctly | The binary target must be derived as `num > 0`; a wrong rule mislabels patients |
| Completeness | Values are present | Cleveland: `ca` missing for 4 patients, `thal` for 2. Hungary: `ca` missing for 291 of 294 |
| Atemporal plausibility | Values are believable | Switzerland: cholesterol is **0** for all 123 patients — impossible, so 0 means "not recorded" |
| Temporal plausibility | Values change sensibly over time | Not checkable: each patient has one visit |

The Swiss cholesterol column is a useful warning. A completeness check that counts blanks would report it as complete. Only a plausibility check reveals that every value is a placeholder.

### 5.3 Quality depends on the purpose

Data quality is not a single score. A dataset is **fit for purpose** if it is good enough for a specific task. The Hungarian data may be adequate for a model that does not use `ca`, but it is unusable for one that does.

**Requirement:** write quality checks as explicit acceptance criteria before modelling — for example, "no input may be missing for more than 10% of patients; any value outside its clinical range is investigated".

## 6. Documentation as a lifecycle output

Decisions made at this stage are easy to forget and hard to reconstruct later. Two documents record them:

- A **datasheet** describes a dataset: why it was collected, who is included, how labels were produced, known gaps, and permitted uses (Gebru et al., 2021).
- A **model card** describes a trained model: its intended use, training data, performance overall and in subgroups, and situations where it should not be used (Mitchell et al., 2019).

For the running example, a datasheet would record the angiography-referred population, the 68% male sample, the 50% narrowing label, and the uncertain timing of `ca`. Later sessions — validation, deployment, and regulation — rely on these records.

## 7. Revision summary

| Concept | Key idea |
|---|---|
| Lifecycle | Framing → data → development → validation → deployment → monitoring → retirement, with loops back |
| Representativeness | Training patients must resemble the intended patients |
| Decision-time availability | Use only inputs that exist when the decision is made |
| Events | Count patients with the outcome, not total rows |
| Labels | State the definition, source, and who could be labelled; beware proxies |
| Data quality | Conformance, completeness, plausibility; verify internally, validate externally |
| Documentation | Datasheets for data, model cards for models |

## 8. Review questions

1. Why does the lifecycle diagram contain a loop from monitoring back to earlier stages? Give one event that could trigger it.
2. Every Cleveland patient had angiography. Explain why this matters for a model intended for routine checkups.
3. A dataset has 5,000 patients and an outcome rate of 2%. How many events does it contain? Why might that be a problem for a model with 40 parameters?
4. A column is complete (no blanks) but every value is 0. Which data-quality category detects the problem, and why does a completeness check miss it?
5. A hospital labels sepsis using billing codes recorded at discharge. Name two ways this label may differ from "the patient had sepsis".
6. Give three facts about the Cleveland data that belong in its datasheet.

## 9. Further reading

- Kahn MG et al. [A harmonized data quality assessment terminology and framework for the secondary use of electronic health record data](https://pmc.ncbi.nlm.nih.gov/articles/PMC5051581/). *eGEMs* 2016;4(1):1244. — the conformance/completeness/plausibility framework used in Section 5.
- Riley RD et al. [Calculating the sample size required for developing a clinical prediction model](https://www.bmj.com/content/368/bmj.m441). *BMJ* 2020;368:m441. — why events, not rows, set the sample size.
- Collins GS et al. [TRIPOD+AI statement](https://www.bmj.com/content/385/bmj-2023-078378). *BMJ* 2024;385:e078378. — 27-item checklist for reporting studies that develop or validate prediction models, including machine-learning models; see also the [TRIPOD website](https://www.tripod-statement.org/).
- US FDA. [Good Machine Learning Practice for medical device development: guiding principles](https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles) — 2021 FDA/Health Canada/MHRA principles and the January 2025 IMDRF final document.
- Gebru T et al. [Datasheets for datasets](https://arxiv.org/abs/1803.09010). *Communications of the ACM* 2021;64(12):86–92.
- Mitchell M et al. [Model cards for model reporting](https://arxiv.org/abs/1810.03993). *FAT\* 2019*.
- UCI Machine Learning Repository. [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) — source files and attribute documentation for the running example.
