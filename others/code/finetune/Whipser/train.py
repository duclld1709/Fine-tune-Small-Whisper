import yaml, os
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from src.data_loader import WhisperDataHandler, DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import WERMetric

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():

    # 1. Loading configs
    cfg = load_config()
    print("Configuration loaded.")

    # 2. Load processor & model
    processor = WhisperProcessor.from_pretrained(cfg['model_name'], language=cfg['language'], task=cfg['task'])
    model = WhisperForConditionalGeneration.from_pretrained(cfg['model_name'])

    model.config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.config.use_cache = False  

    # 3. Prepare Data
    data_handler = WhisperDataHandler(cfg, processor)
    full_dataset = data_handler.load_dataset(from_arrow=True)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric_computer = WERMetric(processor.tokenizer)

    # 4. Training Arguments
    training_args = Seq2SeqTrainingArguments(
            output_dir=cfg['output_dir'],
            per_device_train_batch_size=cfg['batch_size'],
            gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
            learning_rate=float(cfg['learning_rate']),
            warmup_steps=cfg['warmup_steps'],
            max_steps=cfg['max_steps'], 
            fp16=cfg['fp16'],
            eval_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=448,
            save_steps=cfg['save_steps'],
            eval_steps=cfg['eval_steps'],
            logging_steps=cfg['logging_steps'],
            report_to=["wandb"],
            run_name="whisper-finetune-v1",
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            save_total_limit=3,
            push_to_hub=False, 
            disable_tqdm=False,
            ddp_find_unused_parameters=False
        )

    # 5.Initialize Trainer 
    trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=full_dataset["train"],
            eval_dataset=full_dataset["test"],
            data_collator=data_collator,
            compute_metrics=metric_computer.compute_metrics,
            processing_class=processor.feature_extractor,
        )
    
    # 6.Auto-Resume Logic
    last_checkpoint = None
    if os.path.isdir(cfg['output_dir']):
            checkpoints = [d for d in os.listdir(cfg['output_dir']) if d.startswith("checkpoint-")]
            if len(checkpoints) > 0:
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                last_checkpoint = os.path.join(cfg['output_dir'], checkpoints[-1])
                print(f"Found checkpoint: {last_checkpoint}. Resuming training...")
    
    # 7. Start Training
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # 8. Save Final Model
    trainer.save_model(os.path.join(cfg['output_dir'], "final_model"))
    processor.save_pretrained(os.path.join(cfg['output_dir'], "final_model"))

if __name__ == "__main__":
    main()














