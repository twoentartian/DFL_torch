import sys, os, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import py_src.ml_setup_base.dataset_modular as dataset_modular

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate an arithmetic dataset.')
    parser.add_argument("-o", "--operator", type=str, default='+', help=f"operator type, can be one of {dataset_modular.VALID_OPERATORS.keys()}")
    parser.add_argument("-p", "--train_percent", type=int, default=50, help="the ratio of training partition, default=50")
    parser.add_argument("-m", "--modulus", type=int, default=97, help="the modulus, default=97")
    parser.add_argument("-l", "--operand_length", type=int, default=5, help="the length of lists for list operands, default=5")

    args = parser.parse_args()

    # Generate datasets
    train_dataset, val_dataset = dataset_modular.ArithmeticDataset.splits(
        train_pct=args.train_percent,
        operator=args.operator,
        modulus=args.modulus,
        operand_length=args.operand_length,
    )
    assert train_dataset.name == val_dataset.name

    train_file = train_dataset.save_to_file(f"./{train_dataset.name}/train.txt")
    val_file = val_dataset.save_to_file(f"./{val_dataset.name}/val.txt")
    tokenizer_file = train_dataset.tokenizer.save_tokens(f"./{val_dataset.name}/tokenizer.txt")

    print("\nFiles created:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {tokenizer_file} (tokenizer vocabulary)")