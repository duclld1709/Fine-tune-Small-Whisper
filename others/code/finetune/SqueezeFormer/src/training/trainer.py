import torch
import torch.nn as nn
import time
import tqdm
import wandb

def train_epoch(model, dataloader, optimizer, scheduler, scaler, ctc_loss, device, feature_extractor, max_norm=5.0, accum_steps=4, epoch=0):
    model.train()
    total_loss = 0
    
    # Optional SpecAugment
    from ..features.mel_extractor import SpecAugment
    spec_augment = SpecAugment().to(device)
    
    optimizer.zero_grad()
    
    for batch_idx, (waveforms, transcripts, input_lengths, target_lengths) in enumerate(tqdm.tqdm(dataloader, desc="Training")):
        waveforms = waveforms.to(device)
        transcripts = transcripts.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)
        
        with torch.no_grad():
            waveforms, input_lengths = feature_extractor(waveforms, input_lengths)
            # Apply SpecAugment during training
            waveforms = spec_augment(waveforms)
        
        autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=autocast_device_type, enabled=(device.type == "cuda")):
            logits, out_lengths = model(waveforms, input_lengths)
            # CTC loss expects (time, batch, vocab)
            log_probs = nn.functional.log_softmax(logits, dim=-1).transpose(0, 1)
            
            loss = ctc_loss(log_probs, transcripts, out_lengths, target_lengths)
            loss = loss / accum_steps
            
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            # Optional: gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            
            # If scale hasn't decreased, the gradients were not inf/NaN and optimizer.step() was executed
            if scale_before <= scale_after:
                scheduler.step()
                
            optimizer.zero_grad()
            
            # Log to wandb
            wandb.log({
                "train/loss_step": loss.item() * accum_steps,
                "train/lr_step": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else scheduler.optimizer.param_groups[0]['lr'],
                "epoch": epoch
            })
        
        total_loss += loss.item() * accum_steps
        
    # Handle the remaining accumulated gradients if the dataset size isn't a multiple of accum_steps
    if (batch_idx + 1) % accum_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        
        if scale_before <= scale_after:
            scheduler.step()
            
        optimizer.zero_grad()
        
        # Log to wandb for remainder
        wandb.log({
            "train/loss_step": loss.item() * accum_steps,
            "train/lr_step": scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else scheduler.optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })
        
    # Since dataloader is Iterable, we just use batch_idx + 1
    return total_loss / (batch_idx + 1)

def evaluate(
    model,
    dataloader,
    device,
    tokenizer,
    feature_extractor,
    beam_search_decoder=None,
    return_details=False,
    sample_rate=16000,
    progress_desc="Evaluating",
    progress_total=None,
):
    from jiwer import wer, cer
    
    model.eval()
    predictions = []
    references = []
    total_samples = 0
    total_audio_seconds = 0.0
    total_inference_seconds = 0.0
    
    with torch.no_grad():
        for waveforms, transcripts, input_lengths, target_lengths in tqdm.tqdm(
            dataloader, desc=progress_desc, total=progress_total
        ):
            total_samples += waveforms.shape[0]
            total_audio_seconds += input_lengths.sum().item() / float(sample_rate)

            waveforms = waveforms.to(device)
            input_lengths = input_lengths.to(device)
            
            waveforms, input_lengths = feature_extractor(waveforms, input_lengths)
            
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()

            autocast_device_type = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=autocast_device_type, enabled=(device.type == "cuda")):
                logits, out_lengths = model(waveforms, input_lengths)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_inference_seconds += time.perf_counter() - start
                
            # shape: (batch, time, vocab)
            log_probs = nn.functional.log_softmax(logits, dim=-1)
            
            if beam_search_decoder is not None:
                decoded = beam_search_decoder(log_probs, out_lengths)
            else:
                # Greedy Decoding
                preds = log_probs.argmax(dim=-1)
                decoded = []
                for p in preds:
                    # Remove blanks (0) and repeated tokens
                    p = p.tolist()
                    collapsed = []
                    prev = -1
                    for t in p:
                        if t != 0 and t != prev:
                            collapsed.append(t)
                        prev = t
                    decoded.append(collapsed)
            
            # Convert IDs back to text
            for i in range(len(decoded)):
                pred_text = tokenizer.decode(decoded[i])
                
                # Get reference text
                ref_len = target_lengths[i].item()
                ref_ids = transcripts[i, :ref_len].tolist()
                ref_text = tokenizer.decode(ref_ids)
                
                # Post-process Spm special characters
                pred_text = pred_text.replace(' ', ' ').strip()
                ref_text = ref_text.replace(' ', ' ').strip()
                
                predictions.append(pred_text)
                references.append(ref_text)
                
    if references:
        w_error = wer(references, predictions)
        c_error = cer(references, predictions)
    else:
        w_error = 1.0
        c_error = 1.0

    metrics = {
        "wer": w_error,
        "cer": c_error,
        "num_samples": total_samples,
        "audio_seconds": total_audio_seconds,
        "inference_seconds": total_inference_seconds,
        "samples_per_second": (total_samples / total_inference_seconds) if total_inference_seconds > 0 else 0.0,
        "rtf": (total_inference_seconds / total_audio_seconds) if total_audio_seconds > 0 else 0.0,
    }

    if return_details:
        return metrics
    
    return metrics["wer"], metrics["cer"]
