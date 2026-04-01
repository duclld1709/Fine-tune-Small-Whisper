import os
import sentencepiece as spm


def train_sentencepiece_tokenizer_from_texts(texts, vocab_size=2000, model_prefix="unigram"):
    """
    Train SentencePiece tokenizer from an iterable of text lines.
    """
    from ..dataset.normalization import normalize_vietnamese

    temp_txt_path = f"temp_transcripts_{model_prefix}.txt"
    print(f"Extracting transcripts for tokenizer training into {temp_txt_path}...")

    count = 0
    with open(temp_txt_path, "w", encoding="utf-8") as f:
        for transcript in texts:
            if transcript:
                norm_transcript = normalize_vietnamese(transcript)
                if norm_transcript:
                    f.write(norm_transcript + "\n")
                    count += 1
                    if count % 10000 == 0:
                        print(f"Processed {count} transcripts...")

    if count == 0:
        raise ValueError("No valid transcripts found to train tokenizer.")

    print(f"Total extracted transcripts: {count}")
    print(f"Training SentencePiece tokenizer (Unigram, vocab={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=temp_txt_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=[]
    )

    if os.path.exists(temp_txt_path):
        os.remove(temp_txt_path)

    print(f"Tokenizer trained and saved as {model_prefix}.model and {model_prefix}.vocab")


def train_sentencepiece_tokenizer(data_files="Code/src/data/train/*.parquet", vocab_size=2000, model_prefix="vimd_unigram"):
    import glob
    import pyarrow.parquet as pq
    from ..dataset.normalization import normalize_vietnamese
    
    print("Loading ViMD dataset transcripts...")
    files = glob.glob(data_files)
    if not files:
        print(f"Warning: No parquet files found matching {data_files}")
        
    temp_txt_path = "temp_transcripts.txt"
    print("Extracting transcripts for tokenizer training...")
    
    count = 0
    with open(temp_txt_path, "w", encoding="utf-8") as f:
        for filepath in files:
            pf = pq.ParquetFile(filepath)
            # Find the transcript column name
            schema_names = pf.schema.names
            text_col = None
            if "transcript" in schema_names:
                text_col = "transcript"
            elif "text" in schema_names:
                text_col = "text"
            else:
                continue
                
            table = pf.read(columns=[text_col])
            texts = table[text_col].to_pylist()
            
            for transcript in texts:
                if transcript:
                    norm_transcript = normalize_vietnamese(transcript)
                    if norm_transcript:
                        f.write(norm_transcript + "\n")
                        count += 1
                        if count % 10000 == 0:
                            print(f"Processed {count} transcripts...")
                            
    print(f"Total extracted transcripts: {count}")
    print(f"Training SentencePiece tokenizer (Unigram, vocab={vocab_size})...")
    spm.SentencePieceTrainer.train(
        input=temp_txt_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=[]
    )
    
    if os.path.exists(temp_txt_path):
        os.remove(temp_txt_path)
        
    print(f"Tokenizer trained and saved as {model_prefix}.model and {model_prefix}.vocab")

class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
    def encode(self, text):
        return self.sp.encode_as_ids(text)
        
    def decode(self, ids):
        # Convert list of integers or tensor to python list
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.sp.decode_ids(ids)
        
    @property
    def vocab_size(self):
        return self.sp.vocab_size()

if __name__ == "__main__":
    import sys
    data_files = sys.argv[1] if len(sys.argv) > 1 else "Code/src/data/train/*.parquet"
    train_sentencepiece_tokenizer(data_files)
