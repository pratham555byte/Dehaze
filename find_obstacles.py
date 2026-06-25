import os

for root, dirs, files in os.walk('.'):
    # Modify dirs in-place to skip scanning these subdirectories entirely
    dirs[:] = [d for d in dirs if d not in ('.venv', '.git', '__pycache__', 'comparison_results', 'results')]
    for file in files:
        if file.endswith('.py') or file.endswith('.html'):
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'obstacles =' in content or 'obstacles.append' in content:
                    print(f"{path}: matches obstacles pattern")
            except Exception as e:
                pass
