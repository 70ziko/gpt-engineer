import os
import subprocess

from pathlib import Path

import typer

from eval_tools import check_evaluation_component, load_evaluations_from_file, to_emoji
from tabulate import tabulate
from datetime import datetime

from gpt_engineer.db import DB

app = typer.Typer()  # creates a CLI app


def single_evaluate(eval_ob: dict) -> list[bool]:
    """Evaluates a single prompt for creating a new project."""
    print(f"running evaluation: {eval_ob['name']}")

    workspace = DB(eval_ob["project_root"])
    base_abs = Path(os.getcwd())
    code_base_abs = base_abs / eval_ob["project_root"]

    # write to the consent file so we don't get a prompt that hangs
    consent_file = base_abs / ".gpte_consent"
    consent_file.write_text("false")

    # Step 1. Setup known project
    # write the folder and prompt file

    print(f"prompt: {eval_ob['code_prompt']}")
    prompt_path = code_base_abs / "prompt"
    workspace[prompt_path] = f"{eval_ob['code_prompt']}\n"

    # Step 2. Run gpt-engineer
    log_path = code_base_abs / "log.txt"
    log_file = open(log_path, "w")
    process = subprocess.Popen(
        [
            "python",
            "-u",  # Unbuffered output
            "-m",
            "gpt_engineer.main",
            eval_ob["project_root"],
            "--steps",
            "eval_new_code",
            "--temperature",
            "0",
        ],
        stdout=log_file,
        stderr=log_file,
        bufsize=0,
    )
    print(f"waiting for {eval_ob['name']} to finish.")
    process.wait()  # we want to wait until it finishes.

    print("running tests on the newly generated code")
    # TODO: test the code we should have an executable name
    evaluation_results = []
    for test_case in eval_ob["expected_results"]:
        print(f"checking: {test_case['type']}")
        test_case["project_root"] = Path(eval_ob["project_root"])
        evaluation_results.append(check_evaluation_component(test_case))

    return evaluation_results


def generate_report(evals: list[dict], res: list[list[bool]]) -> None:
    pass  # TODO: modify the code from existing code evals.
    # High level shows if all the expected_results passed
    # Detailed shows all the test cases and a pass/fail for each
    output_lines = []
    current_date = datetime.now().strftime("%Y-%m-%d")
    output_lines.append(f"## {current_date}\n\n")

    # Create a summary table
    headers = ["Project", "Evaluation", "All Tests Pass"]
    rows = []
    for i, eval_ob in enumerate(evals):
        rows.append(
            [eval_ob["project_root"], eval_ob["name"], to_emoji(all(res[i]))]
        )  # logical AND of all tests
    table: str = tabulate(rows, headers, tablefmt="pipe")
    title = "New Code Evaluation Summary:"
    print(f"\n{title}\n")
    print(table)
    print()
    output_lines.append(f"### {title}\n\n{table}\n\n")

    # Create a detailed table
    headers = ["Project", "Evaluation", "Test", "Pass"]
    rows = []
    for i, eval_ob in enumerate(evals):
        for j, test in enumerate(eval_ob["expected_results"]):
            rows.append(
                [
                    eval_ob["project_root"],
                    eval_ob["name"],
                    eval_ob["expected_results"][j]["type"],
                    to_emoji(res[i][j]),
                ]
            )
    detail_table: str = tabulate(rows, headers, tablefmt="pipe")
    title = "Detailed Test Results:"
    print(f"\n{title} \n")
    print(detail_table)
    print()

    output_lines.append(f"### {title}\n\n{detail_table}\n\n")
    with open("evals/EVAL_NEW_CODE_RESULTS.md", "a") as file:
        file.writelines(output_lines)


def run_all_evaluations(eval_list: list[dict]) -> None:
    results = []
    for eval_ob in eval_list:
        results.append(single_evaluate(eval_ob))

    # Step 4. Generate Report
    generate_report(eval_list, results)


@app.command()
def main(
    test_file_path: str = typer.Argument("evals/new_code_eval.yaml", help="path"),
):
    if not os.path.isfile(test_file_path):
        raise Exception(f"sorry the file: {test_file_path} does not exist.")

    eval_list = load_evaluations_from_file(test_file_path)
    run_all_evaluations(eval_list)


if __name__ == "__main__":
    app()
