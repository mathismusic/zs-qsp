### Train, and prepare at once! Zero-shot quantum state preparation with deep reinforcement learning

Submission to ML4PS 2024 (NeurIPS 2024 Workshop).

Running instructions (tested with Python 3.11):

1. Install the required packages:
```pip install -r requirements.txt```

2. Run the main script:
```python runner.py -fromjson config.json```
where `config.json` is a JSON file containing all necessary hyper-parameters for the training. Some sample configurations are provided in the `tests` folder.