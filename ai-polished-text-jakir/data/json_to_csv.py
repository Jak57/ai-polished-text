import json
import pandas as pd


def json_to_csv(json_path, csv_path):
    """
    Converts a JSON file to a CSV file.
    """
    with open(json_path, "r") as f:
        d = json.load(f)

    df = pd.DataFrame(d)
    df.to_csv(csv_path, index=False)
    return df


def add_column_to_csv(csv_path, column_name, column_data):
    """
    Adds a column to a CSV file.
    """
    df = pd.read_csv(csv_path)
    df[column_name] = column_data
    df.to_csv(csv_path, index=False)


def rename_column_in_csv(csv_path, old_column_name, new_column_name):
    """
    Renames a column in a CSV file.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={old_column_name: new_column_name})
    df.to_csv(csv_path, index=False)


def reorder_column_in_csv(csv_path, new_column_order):
    """
    Reorders columns in a CSV file.
    """
    df = pd.read_csv(csv_path)
    df = df.reindex(columns=new_column_order)
    df.to_csv(csv_path, index=False)


def main_hwt():
    """
    Load the JSON file
    """
    json_path = "./MixSet/data/selected_pure_data/HWT_original_data.json"
    csv_path = "HWT_original_data.csv"

    df = json_to_csv(json_path, csv_path)
    add_column_to_csv(csv_path, "attack", ["none"] * len(df))
    add_column_to_csv(csv_path, "repetition_penalty", ["none"] * len(df))
    add_column_to_csv(csv_path, "decoding", ["none"] * len(df))
    add_column_to_csv(csv_path, "model", ["human"] * len(df))

    rename_column_in_csv(csv_path, "HWT_sentence", "generation")
    rename_column_in_csv(csv_path, "category", "domain")
    new_column_order = ["id", "model", "decoding", "repetition_penalty", "attack", "domain", "generation"]
    reorder_column_in_csv(csv_path, new_column_order)


def main():
    main_hwt()


if __name__ == "__main__":
    main()

