import os
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "D:/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:/hf_cache/models"

import evaluate

class WERMetric:
    def __init__(self, tokenizer):
        self.metric = evaluate.load("wer")
        self.tokenizer = tokenizer

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    
# Run file to test
if __name__ == "__main__":

    from transformers import WhisperProcessor
    import numpy as np

    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    tokenizer = processor.tokenizer

    wer_metric = WERMetric(tokenizer)

    refs = ["hello world", "machine learning is fun"]
    preds = ["hello word", "machine learning fun"]

    pred_ids = tokenizer(preds, return_tensors="np", padding=True).input_ids
    label_ids = tokenizer(refs, return_tensors="np", padding=True).input_ids
    label_ids[label_ids == tokenizer.pad_token_id] = -100

    class DummyPred:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids

    dummy = DummyPred(pred_ids, label_ids)
    result = wer_metric.compute_metrics(dummy)
    print("WER result:", result)
