"""
Step 6: Merge LoRA adapters and export the final model.

Usage:
    python scripts/06_merge_and_export.py \
        --model_dir ./checkpoints/phase2/final \
        --output_dir ./model_final \
        --save_method merged_16bit
"""

import argparse
from pathlib import Path


def main(args):
    from unsloth import FastModel

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {args.model_dir}")
    model, processor = FastModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=1024,
        load_in_4bit=True,
    )

    print(f"\nMerging LoRA adapters ({args.save_method})...")
    model.save_pretrained_merged(
        str(output_dir),
        processor,
        save_method=args.save_method,
    )

    print(f"\nModel exported to: {output_dir}")
    print(f"Save method: {args.save_method}")

    if args.save_method == "merged_16bit":
        print("\nThis is a full-precision merged model suitable for:")
        print("  - Production deployment")
        print("  - Further fine-tuning")
        print("  - Quantization with other tools")
    elif args.save_method == "merged_4bit":
        print("\nThis is a 4-bit quantized merged model suitable for:")
        print("  - Memory-efficient inference")
        print("  - Edge deployment")

    # Push to HuggingFace Hub (optional)
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_model_id}")
        model.push_to_hub_merged(
            args.hub_model_id,
            processor,
            save_method=args.save_method,
            token=args.hub_token,
        )
        print(f"Model pushed to: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and export fine-tuned model")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./model_final")
    parser.add_argument("--save_method", type=str, default="merged_16bit",
                        choices=["merged_16bit", "merged_4bit", "lora"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)

    args = parser.parse_args()
    main(args)
