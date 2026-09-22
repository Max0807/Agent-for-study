import json
from pathlib import Path

from config import create_client, format_usage, print_api_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_FILE = PROJECT_ROOT / "data" / "route_eval_cases.json"
IMPROVED_PROMPT_FILE = PROJECT_ROOT / "prompts" / "intent_router.txt"

BASELINE_PROMPT = (
    "将问题分类为 knowledge_base、calculator、sql 或 clarify。"
)


def run_prompt(client, model: str, instructions: str, question: str):
    return client.responses.create(
        model=model,
        instructions=instructions,
        input=question,
        max_output_tokens=100,
    )


def main() -> None:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    improved_prompt = IMPROVED_PROMPT_FILE.read_text(encoding="utf-8")

    try:
        client, settings = create_client()
    except Exception as error:
        print_api_error(error)
        raise SystemExit(1) from error

    prompts = {
        "baseline": BASELINE_PROMPT,
        "improved": improved_prompt,
    }

    for prompt_name, prompt in prompts.items():
        passed = 0
        print(f"\n=== {prompt_name} ===")
        for case in cases:
            try:
                response = run_prompt(
                    client,
                    settings.model,
                    prompt,
                    case["question"],
                )
            except Exception as error:
                print_api_error(error)
                raise SystemExit(1) from error

            actual = response.output_text.strip()
            is_passed = actual == case["expected"]
            passed += int(is_passed)
            marker = "PASS" if is_passed else "FAIL"
            print(
                f"{case['id']}: expected={case['expected']}, "
                f"actual={actual!r} -> {marker}"
            )
            print("  " + format_usage(response))

        accuracy = passed / len(cases)
        print(f"Accuracy: {passed}/{len(cases)} = {accuracy:.0%}")


if __name__ == "__main__":
    main()

