import os
import torch
from functools import partial
import glob

from src.dataset.vimd_loader import load_audio_waveform, load_vimd_dataset
from src.tokenizer.train_tokenizer import Tokenizer
from src.features.mel_extractor import LogMelExtractor
from src.models.squeezeformer import SqueezeformerEncoder

def test_sample(checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Configuration
    model_prefix = "vimd_unigram"
    vocab_size = 2000
    
    # Try to find the latest checkpoint if not provided
    if checkpoint_path is None:
        checkpoints = glob.glob("squeezeformer_epoch_*.pt")
        if not checkpoints:
            print("No checkpoints found. Please wait for training to save at least one epoch.")
            return
        # Sort by epoch number
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = checkpoints[-1]
        
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # 2. Load Tokenizer
    if not os.path.exists(f"{model_prefix}.model"):
        print(f"Tokenizer model {model_prefix}.model not found!")
        return
    tokenizer = Tokenizer(f"{model_prefix}.model")
    
    # 3. Model initialization and load weights
    model = SqueezeformerEncoder(
        input_dim=80, num_layers=16, d_model=256, num_heads=4, 
        ffn_dim=1024, conv_kernel=31, dropout=0.0, vocab_size=vocab_size # Set dropout=0 for inference
    ).to(device)
    
    # Handle compiled models weights if necessary
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Remove '_orig_mod.' prefix if model was compiled
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    
    # 4. Feature Extractor
    feature_extractor = LogMelExtractor(
        sample_rate=16000, n_fft=400, hop_length=160, win_length=400, n_mels=80, f_min=0, f_max=8000
    ).to(device)
    
    # 5. Load a sample
    print("Loading a sample from the dataset...")
    # Get a few samples to test
    data_files = "Code/src/data/train/*.parquet"
    dataset = load_vimd_dataset(data_files=data_files, streaming=True)
    
    iterator = iter(dataset)
    
    # Test on the first 3 samples
    for i in range(3):
        try:
            sample = next(iterator)
        except StopIteration:
            break
            
        print(f"\n--- Sample {i+1} ---")
        target_transcript = sample["transcript"]
        print(f"Target:    {target_transcript}")
        
        # Extract audio
        waveform_tensor = load_audio_waveform(sample["audio"], target_sr=16000)

        # Add batch dim
        waveform_tensor = waveform_tensor.unsqueeze(0).to(device)
        input_lengths = torch.tensor([waveform_tensor.shape[1]], dtype=torch.long).to(device)
        
        # Forward Pass
        with torch.no_grad():
            features, feat_lengths = feature_extractor(waveform_tensor, input_lengths)
            
            with torch.amp.autocast(device_type="cuda"):
                logits, out_lengths = model(features, feat_lengths)
                
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Greedy Decoding
            preds = log_probs.argmax(dim=-1)[0] # Take first item in batch
            
            # CTC Decode
            p = preds.tolist()
            collapsed = []
            prev = -1
            for t in p:
                if t != 0 and t != prev: # 0 is blank
                    collapsed.append(t)
                prev = t
                
            predicted_text = tokenizer.decode(collapsed)
            # Post-process Spm special characters
            predicted_text = predicted_text.replace(' ', ' ').strip()
            
            print(f"Predicted: {predicted_text}")

if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    test_sample(checkpoint)
