# script-separation-cvit
Script Identification for Indic Printed Documents – CVIT Project (IIIT-Hyderabad)
# Script: Test ResNet50 on Word Image Dataset

This script loads a trained ResNet50 model and evaluates it on a test dataset of word-level images to identify the script/language (e.g., English, Bengali).

## Features

- Loads a pretrained model.
- Evaluates performance with accuracy and loss.
- Generates a confusion matrix.
- Saves predictions with confidence scores.

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

## Usage

Update the following paths in the script before running:

- `TEST_DIR`: Path to the test dataset.
- `MODEL_WEIGHTS_PATH`: Path to the `.pt` model weights file.

Then run:

```
python your_script_name.py
```

Results will be saved to the `./results/` directory.
