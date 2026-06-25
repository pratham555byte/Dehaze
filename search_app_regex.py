import re

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

keywords = ['twin', 'obstacle', 'digital', 'sim_obstacles']
for i, line in enumerate(lines):
    for kw in keywords:
        if kw in line.lower():
            print(f"Line {i+1}: {line.strip()}")
            break
