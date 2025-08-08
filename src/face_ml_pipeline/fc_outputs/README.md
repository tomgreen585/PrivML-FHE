# Outputs

## CSV File

This is a continuously updated csv file, that updates each time the ML pipeline is run. This helps with keeping track of model performance throughout evaluation.

### Metrics are ordered by

- Timestamp (e.g. when the model was run)
- Model (Model Number)
- Process (Which stage of the ml pipeline)
- Metric (Metric type that is being recorded)
- Score (Metric Score)

## PDF Files

When the ML pipeline is run, a assoicated pdf file is generated for that specific model. This helps with keeping track of model outputs and performance throughout evaluation.

### PDF files contain

- Model number + timestamp
- Model configurations
- Model plots
- Model outputs
