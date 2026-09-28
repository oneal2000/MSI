"""Prepare the shared ToolQA retrieval indexes before multi-GPU evaluation."""

from msi.evaluation.inference import _prewarm_toolqa


def main() -> None:
    _prewarm_toolqa()


if __name__ == "__main__":
    main()
