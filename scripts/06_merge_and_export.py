"""
Step 6: Merge LoRA adapters and export the final model.

Supports both pure HF+PEFT models and Unsloth models.

Usage:
    python scripts/06_merge_and_export.py \
        --model_dir ./checkpoints/phase2/final \
        --output_dir ./model_final \
        --save_method merged_16bit
"""

import argparse
import json
from pathlib import Path

import torch


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    adapter_config_path = model_dir / "adapter_config.json"
    is_peft = adapter_config_path.exists()

    if is_peft:
        # ── Pure HF + PEFT merge path ─────────────────────────────
        print(f"Loading PEFT model from: {args.model_dir}")
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        from peft import PeftModel

        # Read base model name from adapter config
        base_model_name = args.base_model_name
        if base_model_name is None:
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get(
                "base_model_name_or_path",
                "oddadmix/MasriSwitch-Gemma3n-Transcriber-v1"
            )

        print(f"  Base model: {base_model_name}")
        print(f"  Save method: {args.save_method}")

        # Load processor
        try:
            processor = AutoProcessor.from_pretrained(str(model_dir))
        except Exception:
            processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")

        if args.save_method == "lora":
            # Just copy the LoRA adapters
            import shutil
            print("\nCopying LoRA adapters (no merge)...")
            for f in model_dir.iterdir():
                dest = output_dir / f.name
                if f.is_file():
                    shutil.copy2(f, dest)
            processor.save_pretrained(str(output_dir))
            print(f"\nLoRA adapters copied to: {output_dir}")

        else:
            # Merge LoRA into base model
            print(f"\nLoading base model in {'4-bit' if args.save_method == 'merged_4bit' else '16-bit'}...")

            if args.save_method == "merged_4bit":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                base_model = Gemma3nForConditionalGeneration.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation="eager",
                )
            else:
                # merged_16bit: load in full precision for clean merge
                base_model = Gemma3nForConditionalGeneration.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    attn_implementation="eager",
                )

            print("Loading LoRA adapters...")
            model = PeftModel.from_pretrained(base_model, str(model_dir))

            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()

            print(f"Saving merged model to {output_dir}...")
            model.save_pretrained(str(output_dir))
            processor.save_pretrained(str(output_dir))

            print(f"\nModel exported to: {output_dir}")

        if args.save_method == "merged_16bit":
            print("\nThis is a full-precision merged model suitable for:")
            print("  - Production deployment")
            print("  - Further fine-tuning")
            print("  - Quantization with other tools (GGUF, AWQ, etc.)")
        elif args.save_method == "merged_4bit":
            print("\nThis is a 4-bit quantized merged model suitable for:")
            print("  - Memory-efficient inference")
            print("  - Edge deployment")

    else:
        # ── Unsloth merge path (legacy) ───────────────────────────
        print(f"Loading Unsloth model from: {args.model_dir}")
        from unsloth import FastModel

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

    # Push to HuggingFace Hub (optional)
    if args.push_to_hub and args.hub_model_id:
        print(f"\nPushing to HuggingFace Hub: {args.hub_model_id}")
        if is_peft:
            model.push_to_hub(args.hub_model_id, token=args.hub_token)
            processor.push_to_hub(args.hub_model_id, token=args.hub_token)
        else:
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
    parser.add_argument("--base_model_name", type=str, default=None,
                        help="Base model name (auto-detected from adapter_config.json if not set)")
    parser.add_argument("--output_dir", type=str, default="./model_final")
    parser.add_argument("--save_method", type=str, default="merged_16bit",
                        choices=["merged_16bit", "merged_4bit", "lora"])
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)

    args = parser.parse_args()
    main(args)
