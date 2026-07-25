import asyncio
import os
from pathlib import Path

from mutant import augment, load_csv
from mutant.providers.anthropic import AnthropicProvider

async def main() -> None:
    print("🚀 Mutant V0.4 Dataset Augmentation Example")
    
    # 1. Setup the provider
    provider = AnthropicProvider(model="claude-3-5-haiku-20241022")
    
    # 2. Load an existing dataset (CSV, JSON, JSONL, DataFrame, HuggingFace)
    # For this example, we assume we have a simple CSV
    csv_path = Path("sample_dataset.csv")
    if not csv_path.exists():
        with csv_path.open("w") as f:
            f.write("text,intent\n")
            f.write("I need to renew my prescription for Lisinopril.,prescription_renewal\n")
            f.write("What are the side effects of this medication?,medical_query\n")
    
    print(f"Loading {csv_path}...")
    dataset = load_csv(csv_path, text_column="text")
    print(f"Loaded {len(dataset)} scenarios.")
    
    # 3. Augment the dataset with behavioral mutations
    print("\nAugmenting dataset... (this may take a moment)")
    expanded_cases = await augment(
        dataset,
        provider=provider,
        mutations_per_case=3,
        concurrency=5
    )
    
    print(f"\n✅ Augmented into {len(expanded_cases)} behavioral evaluation cases.")
    
    # 4. Export the expanded dataset
    # You can easily export to CSV, JSON, Parquet, Pandas, or HuggingFace format
    out_path = "expanded_dataset.jsonl"
    print(f"Saving to {out_path}...")
    
    with open(out_path, "w", encoding="utf-8") as f:
        for case in expanded_cases:
            f.write(case.model_dump_json() + "\n")
            
    print("Done!")
    
    # Cleanup dummy files
    if csv_path.exists():
        csv_path.unlink()
    if os.path.exists(out_path):
        os.remove(out_path)

if __name__ == "__main__":
    asyncio.run(main())
