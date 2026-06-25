with open('adas_pipeline.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if 'obstacle' in line.lower() or 'dist' in line.lower():
        print(f"Line {i+1}: {line.strip()}")
