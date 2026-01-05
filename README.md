# TwistDAN

🚀 Getting Started
To get the TwistDAN model up and running on your local machine, follow these steps.

📋 Prerequisites
Ensure you have Python installed, along with the following libraries. You can install them by running:

Bash

pip install torch pandas numpy scikit-learn rdkit-pypi
📦 File Structure
TwistDAN.py: The core of the TwistDAN model architecture.

dataprocces.py: Scripts for data loading, preprocessing, and molecular featurization.

train.py: Main script for model training and validation.

test.py: Script to evaluate the performance of a trained model.

TwistDAN_score.py: A command-line utility for fast, single-molecule predictions.

💻 Usage
Training the Model
To train a new TwistDAN model on your dataset, simply run the train.py script. Ensure your data is correctly formatted as a CSV file and the file path is configured in dataprocces.py.

Bash

python train.py
Evaluating Performance
After training, you can evaluate the model's performance on a test set. This will output key metrics such as accuracy, precision, recall, and F1-score.

Bash

python test.py
Fast Prediction
For quick predictions on new molecular structures without a full training pipeline, use the TwistDAN_score.py script.

Bash

The Datasets can be access via https://drive.google.com/drive/folders/1y5Y6RKR5WIsqzHuUo__CNbsegR4_lNQ8?usp=sharing
Bash

python TwistDAN_score.py --input_smiles "CCO"
# Or from a file
python TwistDAN_score.py --input_file molecules.csv --output_file predictions.csv
🌐 Web Server
For instant, no-setup predictions, visit our public web server. The server uses the same core TwistDAN model to provide fast and reliable results.

URL: [https://twistdan.denglab.org/]
