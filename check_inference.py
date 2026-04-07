import torch
import predict

def main():
    predict._load_model()
    tokenizer = predict.tokenizer
    model = predict.model
    if model is None or tokenizer is None:
        raise RuntimeError("Model/tokenizer not loaded. Ensure 'fake-news-bert-base-uncased' exists.")

    text = "Aliens have landed in the park and are giving away free pizza!"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)

    print("Text:", text)
    print("Logits:", logits.tolist())
    print("Probs:", probs.tolist())
    print("Prob(REAL) =", probs[0][0].item(), "Prob(FAKE) =", probs[0][1].item())
    print("predict_news() ->", predict.predict_news(text))

if __name__ == "__main__":
    main()
