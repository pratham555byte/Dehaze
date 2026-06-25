import os

target_dirs = ['.', 'templates']
for d in target_dirs:
    for file in os.listdir(d):
        path = os.path.join(d, file)
        if os.path.isfile(path) and (file.endswith('.py') or file.endswith('.html')):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                if 'obstacles =' in content or 'obstacles.append' in content:
                    print(f"{path}: matches obstacles pattern")
            except Exception as e:
                pass
