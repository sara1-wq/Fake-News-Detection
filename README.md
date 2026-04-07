# Fake-News-Detection
  Technical Stack: Python, Scikit-Learn, Pandas, Streamlit.  
  Analysis: Cleaned and normalized text data to identify sensationalist linguistic patterns often found in "Fake News."  
  Insight: Built a model focused on high precision to ensure reliable news filtering, helping mitigate irrational market movements caused   by viral misinformation.

## Project overview

- `python/app.py`: Streamlit frontend for news classification
- `python/predict.py`: Inference logic using a fine-tuned BERT-based model
- `python/news.csv`: Example dataset with real/fake news labels
- `python/requirements.txt`: Python dependencies
- `python/results/`: Training checkpoints and saved model artifacts

## Requirements

- Python 3.10 or 3.11 recommended
- `pip` package manager

## Install dependencies

```bash
cd python
pip install -r requirements.txt
```

## Run the app

From the `python` folder:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Model location

The app and inference code look for the model in one of these locations:

- `python/fake-news-bert-base-uncased`
- `python/results/checkpoint-<number>`

If your model is not present, place it in one of those paths.

## Notes

- `news.csv` uses `label=1` for fake news and `label=0` for real news.
- The app includes dataset-based overrides so known fake examples from `news.csv` are detected as FAKE.
- Large model files and checkpoint directories should not be committed to GitHub.

## Recommended GitHub workflow

1. Create a new repository.
2. Add this project root and its files.
3. Push to GitHub.
4. Do not include trained model weights or large checkpoints in the repository.

## Optional improvements

- Add a `README.md` summary of results or accuracy metrics
- Add deployment instructions for Streamlit sharing or Heroku
- Add a `python/requirements.txt` with exact package versions

