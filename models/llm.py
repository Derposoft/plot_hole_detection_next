import os
import time
import asyncio
from pathlib import Path, PosixPath
import dotenv
import ast
import json
import uuid
from sklearn.metrics import f1_score, precision_recall_fscore_support
import argparse

from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ospj = os.path.join

REPO_ROOT = Path(__file__).parent.parent
DOTENV_PATH = REPO_ROOT / "secrets" / ".env.local"
dotenv.load_dotenv(DOTENV_PATH)

TEST_DATA_DIR_1_ERROR = REPO_ROOT / "data" / "dataset" / "1_error" / "test"
TEST_DATA_DIR_2_ERROR = REPO_ROOT / "data" / "dataset" / "2_error" / "test"
TEST_DATA_DIR_5_ERROR = REPO_ROOT / "data" / "dataset" / "5_error" / "test"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Don't think we should be using this one, the number of output
# tokens is too variable before we get an actual response.
ANTHROPIC_MODEL_ID_THINKING = "claude-sonnet-4-20250514"
ANTHROPIC_MODEL_ID = "claude-3-5-haiku-20241022"

SYSTEM_PROMPT = """
PROBLEM DESCRIPTION:

You are an expert at finding plot holes in short stories.
You will be given a document and you will need to identify the plot holes in the document.
You will need to return the plot holes in the document as an array of integers.
There will be no more than 5 plot holes in the document.

IMPORTANT:
- Keep in mind that the list of integers you return MUST be 0-indexed.
- Return only the array of integers, no other text.

EXAMPLE:

Example of an input document:

The world is flat.
The sky is blue.
The sea is blue.
The earth is round.
The sky is not blue.

Example response:
[3, 4]
(End of response)

Explanation for why the response is [3, 4]:

In the story's first sentence, the author claims that the world is flat.
In the story's second sentence, the author claims that the sky is blue.
In the story's third sentence, the author claims that the sea is blue.
In the story's fourth sentence, the author claims that the earth is round. This is a plot hole because the earth is actually flat in the author's story.
In the story's fifth sentence, the author claims that the sky is not blue. This is a plot hole because the sky is actually blue in the author's story.

Since sentences are 0-indexed, the response is [3, 4].
"""


def get_anthropic_client() -> Anthropic:
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return client


def submit_batch(document_paths: list[str]) -> str:
    custom_id_path_map = {path: str(uuid.uuid4()) for path in document_paths}
    documents = {}
    for path in document_paths:
        with open(path, "r") as f:
            documents[path] = f.read()
    requests = [
        Request(
            custom_id=custom_id_path_map[path],
            params=MessageCreateParamsNonStreaming(
                model=ANTHROPIC_MODEL_ID,
                max_tokens=20,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": document,
                    },
                ],
            ),
        )
        for path, document in documents.items()
    ]

    client = get_anthropic_client()
    message_batch = client.messages.batches.create(requests=requests)
    return message_batch.id, custom_id_path_map


def is_batch_completed(batch_id: str) -> bool | None:
    client = get_anthropic_client()

    try:
        message_batch = client.messages.batches.retrieve(
            batch_id,
        )
        print(
            f"Batch {message_batch.id} processing status is {message_batch.processing_status}"
        )
        return message_batch.processing_status == "ended"
    except Exception:
        # On rate limiting failures just return None
        return None


class Result:
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    EXPIRED = "expired"


def extract_prediction_from_result(result: str) -> list[int]:
    """
    Gets the list of plot holes from a string result that is [int, int, ...]
    """
    result_first_line = result.splitlines()[0].strip()
    try:
        result = ast.literal_eval(result_first_line)
    except Exception:
        print(f"Invalid result from model: {result}")
        return []

    if not isinstance(result, list):
        print(f"Invalid result from model: {result}")
        return []

    if not all(isinstance(i, int) for i in result):
        print(f"Invalid result from model: {result}")
        return []

    return result


def parse_batch_results(
    batch_id: str, custom_id_path_map: dict[str, str]
) -> dict[str, list[int]]:
    inverse_custom_id_path_map = {v: k for k, v in custom_id_path_map.items()}

    # Parse results and store a map from document path -> prediction
    results = {}
    client = get_anthropic_client()
    for result in client.messages.batches.results(batch_id):
        document_path = inverse_custom_id_path_map[result.custom_id]
        match result.result.type:
            case Result.SUCCEEDED:
                response = result.result.message.content[-1].text
                prediction = extract_prediction_from_result(response)
                results[document_path] = prediction
            case Result.ERRORED:
                print(f"Errored: {result.result.error.type}")
                results[document_path] = []
            case Result.EXPIRED:
                print(f"Expired: {result.result.error.type}")
                results[document_path] = []

    return results


def save_batch_results(results: dict[str, list[int]], output_path: str):
    with open(output_path, "w") as f:
        json.dump(results, f)


def save_custom_id_path_map(custom_id_path_map: dict[str, str], output_path: str):
    with open(output_path, "w") as f:
        json.dump(custom_id_path_map, f)


def get_results(
    documents_dir: PosixPath, dry_run: bool = False
) -> dict[str, list[int]]:
    results_dir = ospj(REPO_ROOT, "results", "llm")
    documents_dir_name = str(documents_dir).replace("/", "-")
    document_paths = [ospj(documents_dir, f) for f in os.listdir(documents_dir)]
    document_paths = [
        x for x in document_paths if x.endswith(".txt") and os.path.isfile(x)
    ]
    if dry_run:
        document_paths = document_paths[:10]

    # Submit batch
    batch_id, custom_id_path_map = submit_batch(document_paths)
    custom_id_path_map_file_name = f"{documents_dir_name}_custom_id_map.json"
    save_custom_id_path_map(
        custom_id_path_map, ospj(results_dir, custom_id_path_map_file_name)
    )
    print(f"Batch {batch_id} submitted")

    # Wait for batch to complete
    completed = is_batch_completed(batch_id)
    wait_time = 30
    while not completed:
        # Wait for batch and exponentially back off if we're being rate limited
        print(f"Batch {batch_id} not completed. Waiting for {wait_time} seconds...")
        if completed is None:
            print(
                "Batch is likely being rate limited while checking for completion! Backing off..."
            )
            wait_time *= 2
        time.sleep(wait_time)

        # Check for completion again
        completed = is_batch_completed(batch_id)

    # Parse, save, etc
    print(f"Batch {batch_id} completed! Parsing results...")
    results = parse_batch_results(batch_id, custom_id_path_map)

    # Save results in case the rest of this script is screwed up
    results_dir = ospj(REPO_ROOT, "results", "llm")
    os.makedirs(results_dir, exist_ok=True)
    results_file_name = f"{documents_dir_name}_{time.time()}.json"
    save_batch_results(results, ospj(results_dir, results_file_name))
    return results


def get_ground_truth_indices(document_path: str) -> list[int]:
    with open(document_path, "r") as f:
        document_first_line = f.readlines()[0].strip()
    # synthetic data is generated with the following format:
    # continuity [1, 2, 3] for legacy reasons
    document_first_line_indices = document_first_line.split("continuity")[1].strip()
    return ast.literal_eval(document_first_line_indices)


def print_metrics(results: dict[str, list[int]]):
    """
    results: {document_path: [plot_hole_indices]}
    """
    # Get the ground truth for each document
    ground_truth_indices = {
        path: get_ground_truth_indices(path) for path in results.keys()
    }

    # Build binary labels across all sentences and use sklearn for metrics
    all_true: list[int] = []
    all_pred: list[int] = []

    for path, predicted_indices in results.items():
        try:
            with open(path, "r") as f:
                lines = f.read().splitlines()
        except Exception:
            continue

        num_sentences = max(0, len(lines) - 1)
        gt_list = ground_truth_indices.get(path, [])

        gt_set = {i for i in (gt_list or []) if 0 <= i < num_sentences}
        pred_set = {i for i in (predicted_indices or []) if 0 <= i < num_sentences}

        y_true_doc = [1 if i in gt_set else 0 for i in range(num_sentences)]
        y_pred_doc = [1 if i in pred_set else 0 for i in range(num_sentences)]

        all_true.extend(y_true_doc)
        all_pred.extend(y_pred_doc)

    if not all_true:
        print("No data to compute metrics.")
        return

    f1_t = f1_score(all_true, all_pred)
    prec_ma, rec_ma, f1_ma, _ = precision_recall_fscore_support(
        all_true, all_pred, average="macro", zero_division=0
    )
    prec_mi, rec_mi, f1_mi, _ = precision_recall_fscore_support(
        all_true, all_pred, average="micro", zero_division=0
    )

    print(
        f"F1-T: {f1_t:.4f} "
        f"P-Ma: {prec_ma:.4f} P-Mi: {prec_mi:.4f} "
        f"R-Ma: {rec_ma:.4f} R-Mi: {rec_mi:.4f} "
        f"F1-Ma: {f1_ma:.4f} F1-Mi: {f1_mi:.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dry_run = args.dry_run

    async def _run():
        results_1_error, results_2_error, results_5_error = await asyncio.gather(
            asyncio.to_thread(get_results, TEST_DATA_DIR_1_ERROR, dry_run),
            asyncio.to_thread(get_results, TEST_DATA_DIR_2_ERROR, dry_run),
            asyncio.to_thread(get_results, TEST_DATA_DIR_5_ERROR, dry_run),
        )

        print("1-error metrics:")
        print_metrics(results_1_error)
        print("2-error metrics:")
        print_metrics(results_2_error)
        print("5-error metrics:")
        print_metrics(results_5_error)

    asyncio.run(_run())
