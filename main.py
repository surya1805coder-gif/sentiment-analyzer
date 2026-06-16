"""
Entry point for training the sentiment analysis model.
Usage:
    python main.py                          # train with defaults
    python main.py --epochs 5 --batch-size 16
"""

import argparse
import sys
from src import (
    get_config,
    load_and_split,
    tokenize_texts,
    to_dataset,
    build_model,
    build_training_args,
    create_trainer,
    evaluate,
    generate_classification_report,
    save_confusion_matrix,
    setup_logging,
    set_seed,
    save_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentiment analysis")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--model-name", type=str, default=None, help="Override model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    cfg = get_config()

    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.model_name:
        cfg["model"]["name"] = args.model_name

    logger = sys.modules["src"].__dict__.get("logging", __import__("logging")).getLogger(__name__)
    logger.info("Configuration: %s", cfg)

    # 1. Load data
    train_texts, val_texts, train_labels, val_labels = load_and_split(
        source=cfg["data"]["url"],
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    # 2. Build model
    model, tokenizer = build_model(
        model_name=cfg["model"]["name"],
        num_labels=cfg["model"]["num_labels"],
        device=cfg["device"],
        cache_dir=cfg["model"]["cache_dir"],
    )

    # 3. Tokenize
    train_encodings = tokenize_texts(tokenizer, train_texts, cfg["data"]["max_length"])
    val_encodings = tokenize_texts(tokenizer, val_texts, cfg["data"]["max_length"])

    train_dataset = to_dataset(train_encodings, train_labels)
    val_dataset = to_dataset(val_encodings, val_labels)

    # 4. Train
    training_args = build_training_args(
        output_dir=cfg["training"]["output_dir"],
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        warmup_steps=cfg["training"]["warmup_steps"],
        fp16=cfg["training"]["fp16"],
        logging_steps=cfg["training"]["logging_steps"],
        save_strategy=cfg["training"]["save_strategy"],
        eval_strategy=cfg["training"]["eval_strategy"],
        load_best=cfg["training"]["load_best"],
        metric_for_best=cfg["training"]["metric_for_best"],
    )

    trainer = create_trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=training_args,
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
    )

    trainer.train()

    # 5. Evaluate
    metrics = evaluate(trainer, cfg["training"]["output_dir"])
    generate_classification_report(
        trainer,
        val_dataset,
        label_names=["negative", "neutral", "positive"],
        output_dir=cfg["training"]["output_dir"],
    )
    save_confusion_matrix(
        trainer,
        val_dataset,
        label_names=["negative", "neutral", "positive"],
        output_dir=cfg["training"]["output_dir"],
    )

    # 6. Save
    model_dir = f"{cfg['training']['output_dir']}/sentiment_model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    save_metadata(model_dir, metrics, cfg)

    logger.info("Training complete. Model saved to %s", model_dir)


if __name__ == "__main__":
    main()
