import json
import pandas as pd


def json_to_csv(src_path, dest_path):
    """
    Convert json file to csv file
    """

    with open(src_path, "r", encoding="utf-8") as f:
        mix_set_hwt = json.load(f)

    id = []
    model = []
    decoding = []
    repetition_penalty = []
    attack = []
    domain = []
    generation = []

    for dic in mix_set_hwt:
        id.append(dic['id'])
        model.append("human")
        decoding.append("none")
        repetition_penalty.append("none")
        attack.append("none")
        domain.append(dic['category'])
        generation.append(dic['HWT_sentence'])

    dataset_dic = {
        'id': id, 
        'model': model, 
        'decoding': decoding, 
        'repetition_penalty': repetition_penalty, 
        'attack': attack, 
        'domain': domain, 
        'generation': generation
    }
    df = pd.DataFrame(dataset_dic, columns=dataset_dic.keys())
    df.to_csv(dest_path, index=False)

if __name__ == "__main__":
    src_path = "./MixSet/data/selected_pure_data/HWT_original_data.json"
    dest_path = "HWT_original_data.csv"
    
    json_to_csv(src_path, dest_path)

