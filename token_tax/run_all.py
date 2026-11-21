"""Command-line entry point for Token Tax."""
import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load_component(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load component {filename}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("bench", "pipeline", "vocab", "native-noise", "dashboard", "all"),
    )
    args = parser.parse_args()

    if args.command == "dashboard":
        _load_component("4_dashboard.py", "token_tax_dashboard").main()
        return

    components = {
        "bench": ("1_fertility_benchmarker.py", "token_tax_bench"),
        "pipeline": ("2_transliteration_pipeline.py", "token_tax_pipeline"),
        "vocab": ("3_vocab_expansion_demo.py", "token_tax_vocab"),
        "native-noise": ("native_noise_experiment.py", "token_tax_native_noise"),
    }
    commands = ("bench", "pipeline", "vocab") if args.command == "all" else (args.command,)
    for command in commands:
        filename, module_name = components[command]
        component = _load_component(filename, module_name)
        component.main()


if __name__ == "__main__":
    main()
