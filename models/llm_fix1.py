import json
import time
from pathlib import Path

import models.llm as llm


REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "llm"
OLD_DIR = RESULTS_DIR / "old"

# Input JSONL file with batch results
JSONL_FILENAME = "5err-msgbatch_01SwS6CpXDeTdU4tynRySY1J_results.jsonl"
JSONL_PATH = RESULTS_DIR / JSONL_FILENAME

# Custom ID map produced when the batch was submitted
CUSTOM_MAP_FILENAME = "-home-derposoft-code-plot_hole_detection_next-data-dataset-5_error-test_custom_id_map.json"
CUSTOM_MAP_PATH = OLD_DIR / CUSTOM_MAP_FILENAME


def main():
    with open(CUSTOM_MAP_PATH, "r") as f:
        path_to_custom_id = json.load(f)
    custom_id_to_path = {v: k for k, v in path_to_custom_id.items()}

    results = {}
    with open(JSONL_PATH, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            custom_id = obj.get("custom_id")
            res = obj.get("result", {})
            result_type = res.get("type")

            if result_type == llm.Result.SUCCEEDED:
                message = res.get("message", {})
                content = message.get("content", [])
                text = ""
                if content:
                    last = content[-1]
                    if isinstance(last, dict):
                        text = last.get("text", "")
                prediction = llm.extract_prediction_from_result(text)
            else:
                prediction = []

            document_path = custom_id_to_path.get(custom_id)
            if document_path is None:
                continue
            results[document_path] = prediction

    documents_dir = REPO_ROOT / "data" / "dataset" / "5_error" / "test"
    documents_dir_name = str(documents_dir).replace("/", "-")
    output_path = OLD_DIR / f"{documents_dir_name}_{time.time()}.json"

    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
