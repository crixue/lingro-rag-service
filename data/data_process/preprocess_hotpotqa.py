import json

from sympy.polys.polyconfig import query

from rag_checker.rag_evaluate import RAGTestItem


def convert_to_jsonl(raw_json_file_path, save_path):
    with open(raw_json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    with open(save_path, "w", encoding="utf-8") as f:
        for key, value in json_data.items():
            f.write(json.dumps({
                key: value
            }) + "\n")


def process_hotpotqa_to_standard_format(query_file_path, save_path):
    '''
    convert hotpotqa queries to RAGTestItem format. The file should be a json file. example:
        [{
        "supporting_facts": [
            [
                "Al\u00fb",
                3
            ],
            [
                "Lilu (mythology)",
                0
            ]
        ],
        "level": "easy",
        "question": "If Gallu is a demon Lilu is what?",
        "context": [
            [
                "Demon Dice",
                [
                    "Demon Dice, originally published as Chaos Progenitus, is a collectible dice game created by Lester Smith (designer of the better-known Dragon Dice) and Tim Brown.",
                    " In it, each player controls a demon made of 13 dice representing body parts (or magical items wielded by those parts), which battles the other player's demon.",
                    " The original Chaos Progenitus version presented the concept of these dueling fiends seriously; when the game was republished as Demon Dice by Fast Forward Games (which has since gone out of business), the idea was given a humorous spin, styling the players as \"Demon Creation Specialists, Junior Grade\" in the bureaucracy of Hell.",
                    " Neither version of the game was a great commercial success, though they retain a small and loyal fanbase."
                ]
            ]
        ],
        "answer": "a spirit",
        "_id": "5a77ec115542992a6e59dff7",
        "type": "bridge"
    }]
    '''
    print("Processing HotpotQA data...")
    hotpotqa_queries = json.load(open(query_file_path))

    def generate_items(queries):
        for item in queries:
            std_item = RAGTestItem(
                query_id=item["_id"],
                query=item["question"],
                gt_answer=item["answer"],
                level=item["level"],
            )
            yield std_item

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for item in generate_items(hotpotqa_queries):
            if not first:
                f.write(",\n")
            json.dump({
                k: v for k, v in item.model_dump().items() if v is not None
            }, f, ensure_ascii=False, indent=4)
            first = False
        f.write("\n]")


if __name__ == "__main__":
    process_hotpotqa_to_standard_format("../raw_data/hotpotqa_train.json", "../processed_data/hotpotqa_train.json")
    convert_to_jsonl("../raw_data/hotpotqa_train_corpus.json", "../processed_data/hotpotqa_train_corpus.jsonl")
