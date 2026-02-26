#!/usr/bin/env python3
"""
Check line length for Python files
"""

import os
import sys
from pathlib import Path

MAX_LINE_LENGTH = 100
EXCLUDE_DIRS = {'.git', '.github', '.tox', '.venv', 'build', 'dist', '_build', '__pycache__'}

def check_line_length(file_path: Path) -> list:
    """Check line length in a Python file"""
    violations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                stripped_line = line.rstrip()
                if len(stripped_line) > MAX_LINE_LENGTH:
                    violations.append((i, len(stripped_line), stripped_line[:50] + '...'))
    except (IOError, UnicodeDecodeError):
        pass
    return violations

def main():
    """Main function"""
    violations = []
    python_files = []

    # Find all Python files
    for root, dirs, files in os.walk('.'):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)

    # Check each file
    for file_path in python_files:
        file_violations = check_line_length(file_path)
        if file_violations:
            for line_num, length, preview in file_violations:
                violations.append((str(file_path), line_num, length, preview))

    if violations:
        print(f'Found {len(violations)} lines exceeding {MAX_LINE_LENGTH} characters:')
        print()
        for file_path, line_num, length, preview in violations[:20]:
            print(f'  {file_path}:{line_num}:{length}')
            print(f'    {preview}')
            print()
        if len(violations) > 20:
            print(f'  ... and {len(violations) - 20} more violations')
        sys.exit(1)
    else:
        print('All lines are within the maximum length limit.')
        print(f'Checked {len(python_files)} Python files.')
        sys.exit(0)

if __name__ == '__main__':
    main()
