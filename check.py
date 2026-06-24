# project_context_generator.py
# Run this file inside your project folder:
# python project_context_generator.py
#
# It will generate:
# project_context.txt
#
# Share that text file with ChatGPT.

import os
import platform
from datetime import datetime

OUTPUT_FILE = "project_context.txt"

# File extensions to scan
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".cpp", ".c", ".h", ".hpp",
    ".java", ".cs",
    ".html", ".css", ".scss",
    ".json", ".yaml", ".yml",
    ".md", ".txt",
    ".ipynb"
}

# Folders to ignore
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "build",
    "dist",
    ".idea",
    ".vscode",
    "outputs",
    "logs",
    "checkpoints",
    "weights"
}

MAX_FILE_PREVIEW_LINES = 40
MAX_TOTAL_FILES = 200


def get_project_tree(root_dir):
    tree = []

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        level = root.replace(root_dir, "").count(os.sep)
        indent = "│   " * level
        folder_name = os.path.basename(root)

        tree.append(f"{indent}├── {folder_name}/")

        sub_indent = "│   " * (level + 1)

        for file in files:
            tree.append(f"{sub_indent}├── {file}")

    return "\n".join(tree)


def read_file_preview(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        preview = "".join(lines[:MAX_FILE_PREVIEW_LINES])

        return preview.strip()

    except Exception as e:
        return f"[Could not read file: {e}]"


def collect_project_info(root_dir):
    info = []

    info.append("=" * 80)
    info.append("PROJECT CONTEXT REPORT")
    info.append("=" * 80)

    info.append(f"\nGenerated: {datetime.now()}")
    info.append(f"Operating System: {platform.system()} {platform.release()}")
    info.append(f"Python Version: {platform.python_version()}")

    info.append("\n" + "=" * 80)
    info.append("PROJECT STRUCTURE")
    info.append("=" * 80)

    info.append(get_project_tree(root_dir))

    info.append("\n" + "=" * 80)
    info.append("IMPORTANT FILE PREVIEWS")
    info.append("=" * 80)

    file_count = 0

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file_count >= MAX_TOTAL_FILES:
                break

            ext = os.path.splitext(file)[1]

            if ext.lower() in CODE_EXTENSIONS:
                path = os.path.join(root, file)

                relative_path = os.path.relpath(path, root_dir)

                info.append("\n" + "-" * 80)
                info.append(f"FILE: {relative_path}")
                info.append("-" * 80)

                preview = read_file_preview(path)

                info.append(preview)

                file_count += 1

    info.append("\n" + "=" * 80)
    info.append("MANUAL QUESTIONS FOR PROJECT UNDERSTANDING")
    info.append("=" * 80)

    questions = [
        "1. What is the main goal of the project?",
        "2. What problem does it solve?",
        "3. What stage is the project currently in?",
        "4. What features are already completed?",
        "5. What features are pending?",
        "6. What tech stack are you using?",
        "7. Which AI/ML models are used?",
        "8. What are the biggest current problems or blockers?",
        "9. What output are you expecting from the system?",
        "10. What help do you need right now?",
        "11. Are there any deadlines, competitions, or hackathons?",
        "12. Any GitHub repositories or references being used?",
        "13. Any performance issues?",
        "14. Any dataset details?",
        "15. Deployment target? (web/app/local/cloud)"
    ]

    info.extend(questions)

    return "\n".join(info)


def main():
    root_dir = os.getcwd()

    print("\nScanning project...")
    print(f"Project Directory: {root_dir}")

    report = collect_project_info(root_dir)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nDone!")
    print(f"Generated file: {OUTPUT_FILE}")
    print("\nShare the contents of this file with ChatGPT.")


if __name__ == "__main__":
    main()