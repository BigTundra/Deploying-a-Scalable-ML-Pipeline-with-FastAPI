# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Developer: Cullen Littlefield
Model Org: WGU / Udacity
Model Version: v1.0.0

Model Type: The base model for this is an AdaBoost Classifier.

## Intended Use
The intended use of this model is to ingest data following the same schema as standard US census data, and predict whether or not the record makes a salary above $50k or $50k and below. applications outside the US or using non-US based census data, may require retraining or have less than desireable results.

## Training Data
The training dataset was as an 80% subset of an overall dataset originally containing 32561 total records. This 80% subset contained 26049 records composed of the following fields, and data-type values.

#   Column            Dtype 
---  ------           ----- 
 0   age              int64 
 1   workclass        object
 2   fnlgt            int64 
 3   education        object
 4   education-num    int64 
 5   marital-status   object
 6   occupation       object
 7   relationship     object
 8   race             object
 9   sex              object
 10  capital-gain     int64 
 11  capital-loss     int64 
 12  hours-per-week   int64 
 13  native-country   object
 14  salary           object

## Evaluation Data

The evaluation dataset was as an 20% subset of an overall dataset originally containing 32561 total records. This 20% subset contained 6512 records composed of the following fields, and data-type values.

#   Column            Dtype 
---  ------           ----- 
 0   age              int64 
 1   workclass        object
 2   fnlgt            int64 
 3   education        object
 4   education-num    int64 
 5   marital-status   object
 6   occupation       object
 7   relationship     object
 8   race             object
 9   sex              object
 10  capital-gain     int64 
 11  capital-loss     int64 
 12  hours-per-week   int64 
 13  native-country   object
 14  salary           object

Evaluation on data sliced categorically was also completed, and an analysis of important evaluation and implementation concerns regarding specific slices is included below in the section titled "Ethical Considerations, Caveats, and Recommendations"

## Metrics
Performance Metrics for this model on the evaluation data include Precision, Recall, and F1. The following are the model metrics after hyperparameter tuning.
Precision: 0.7722 
Recall: 0.6671 
F1: 0.7158

## Ethical Considerations, Caveats, and Recommendations
| Feature            | Concerning slices (low F1 < 0.55 or very unbalanced P/R)                                                                                                                | Strong slices (F1 ≥ 0.75 and reasonably large n)               | Key take-aways & deployment considerations                                                                                                                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **workclass**      | “?” (0.47) • Other-service (0.41)                                                                                                                                       | Self-emp-inc (0.81) • Federal-gov (0.75)                       | Model struggles when *workclass* is missing and for “other-service” jobs -> consider better missing-value handling and adding job-type signals.                                                                                   |
| **education**      | ≤ 8th-grade slices show F1 ≤ 0.18–0.50, driven by recall < 0.35                                                                                                         | Doctorate 0.91 • Masters 0.83 • Prof-school 0.91               | Clear monotonic trend: higher education → higher F1.  Low-education groups are systematically under-detected (low recall).  Flag for fairness monitoring; you may need separate thresholds or re-weighting.                      |
| **marital-status** | Divorced 0.55 • Never-married 0.60 • Married-spouse-absent 0.57                                                                                                         | Married-civ-spouse 0.73                                        | Married individuals benefit from larger data & clearer income patterns; single/divorced groups are mis-classified more often.                                                                                                    |
| **occupation**     | Other-service 0.41 • Transport-moving 0.45 • Machine-op-inspct 0.47 • Craft-repair 0.47                                                                                 | Exec-managerial 0.82 • Prof-specialty 0.81 • Tech-support 0.75 | “White-collar” jobs predict well; manual/service roles have both lower precision *and* recall.  May reflect training-data imbalance—collect more examples or engineer occupation-specific features.                              |
| **relationship**   | Own-child 0.38 • Other-relative 0.44 • Unmarried 0.50                                                                                                                   | Husband 0.73 • Wife 0.74                                       | The model is weakest for dependents (“own-child”) and non-nuclear family relations -> reconsider how age/dependence interacts with income.                                                                                        |
| **race**           | “Other” (precision 1.00, recall 0) -> F1 0.00                                                                                                                            | White 0.72                                                     | Tiny support for “Other” gives unstable metrics; data augmentation or grouping rare races may be needed.  Black and Asian slices are \~0.65–0.70 (10 pp lower than Whites).  Monitor for racial fairness.                        |
| **sex**            | —                                                                                                                                                                       | Male 0.72 vs Female 0.66                                       | 6 pp F1 gap; recall for females lags by \~7 pp.  Consider threshold tuning by sex or capturing additional features that correlate with female income patterns (e.g., occupation breaks).                                         |
| **native-country** | Several small-n countries show *either* F1 0 (e.g., Vietnam, South) or 1.0 (Dominican-Republic) due to ≤ 3 positives; Mexico 0.40–0.53 despite n≈130; “?” group F1 0.70 | United-States 0.72 (n ≈ 5800)                                  | Very sparse tail of countries causes extreme scores.  You likely need to bucket low-frequency countries or use region-level encoding.  Monitor U.S. vs non-U.S. disparity; Mexican-born individuals are notably under-predicted. |