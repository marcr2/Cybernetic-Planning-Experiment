#!/usr / bin / env python3
"""
Code Cleanup Script

Performs comprehensive code cleanup including:
- Removing unused imports - Fixing code formatting - Removing dead code - Optimizing imports - Cleaning up temporary files
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Set, Dict, Any
import subprocess

class CodeCleanup:
    """Comprehensive code cleanup utility."""

    def __init__(self, project_root: str = "."):
        """
        Initialize the code cleanup utility.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.python_files = []
        self.cleanup_stats = {
            "files_processed": 0,
            "unused_imports_removed": 0,
            "dead_code_removed": 0,
            "formatting_fixes": 0,
            "files_cleaned": 0
        }

    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []

        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules'}]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        self.python_files = python_files
        return python_files

    def remove_unused_imports(self, file_path: Path) -> bool:
        """
        Remove unused imports from a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            True if changes were made
        """
        try:
            with open(file_path, 'r', encoding='utf - 8') as f:
                content = f.read()

            # Parse the AST
            tree = ast.parse(content)

            # Find all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")

            # Find all names used in the code
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    # Handle cases like module.function
                    if isinstance(node.value, ast.Name):
                        used_names.add(node.value.id)

            # Remove unused imports
            lines = content.split('\n')
            new_lines = []
            removed_imports = 0

            for line in lines:
                # Check if this is an import line
                if line.strip().startswith(('import ', 'from ')):
                    # Extract the imported name
                    import_match = re.match(r'^(?:from\s+\S+\s+)?import\s+(.+)$', line.strip())
                    if import_match:
                        import_names = import_match.group(1).split(',')
                        import_names = [name.strip().split(' as ')[0].split('.')[0] for name in import_names]

                        # Check if any of the imported names are used
                        if not any(name in used_names for name in import_names):
                            removed_imports += 1
                            continue  # Skip this line

                new_lines.append(line)

            if removed_imports > 0:
                with open(file_path, 'w', encoding='utf - 8') as f:
                    f.write('\n'.join(new_lines))

                self.cleanup_stats["unused_imports_removed"] += removed_imports
                return True

            return False

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False

    def fix_code_formatting(self, file_path: Path) -> bool:
        """
        Fix basic code formatting issues.

        Args:
            file_path: Path to the Python file

        Returns:
            True if changes were made
        """
        try:
            with open(file_path, 'r', encoding='utf - 8') as f:
                content = f.read()

            original_content = content

            # Fix common formatting issues
            # Remove trailing whitespace
            content = re.sub(r'[ \t]+$', '', content, flags = re.MULTILINE)

            # Ensure exactly one newline at end of file
            content = content.rstrip() + '\n'

            # Fix multiple blank lines (max 2)
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

            # Fix spaces around operators
            content = re.sub(r'(\w)\s*=\s*(\w)', r'\1 = \2', content)
            content = re.sub(r'(\w)\s*\+\s*(\w)', r'\1 + \2', content)
            content = re.sub(r'(\w)\s*-\s*(\w)', r'\1 - \2', content)
            content = re.sub(r'(\w)\s*\*\s*(\w)', r'\1 * \2', content)
            content = re.sub(r'(\w)\s*/\s*(\w)', r'\1 / \2', content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf - 8') as f:
                    f.write(content)

                self.cleanup_stats["formatting_fixes"] += 1
                return True

            return False

        except Exception as e:
            print(f"Error formatting {file_path}: {e}")
            return False

    def remove_dead_code(self, file_path: Path) -> bool:
        """
        Remove obvious dead code.

        Args:
            file_path: Path to the Python file

        Returns:
            True if changes were made
        """
        try:
            with open(file_path, 'r', encoding='utf - 8') as f:
                content = f.read()

            original_content = content

            # Remove commented - out code blocks
            lines = content.split('\n')
            new_lines = []
            in_comment_block = False

            for line in lines:
                stripped = line.strip()

                # Skip lines that are just comments
                if stripped.startswith('#') and not stripped.startswith('# '):
                    # Check if it looks like commented - out code
                    if any(keyword in stripped.lower() for keyword in ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ']):
                        continue  # Skip this line

                new_lines.append(line)

            content = '\n'.join(new_lines)

            # Remove empty functions and classes
            content = re.sub(r'def\s+\w+\([^)]*\):\s*\n\s * pass\s*\n', '', content)
            content = re.sub(r'class\s+\w+[^:]*:\s*\n\s * pass\s*\n', '', content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf - 8') as f:
                    f.write(content)

                self.cleanup_stats["dead_code_removed"] += 1
                return True

            return False

        except Exception as e:
            print(f"Error removing dead code from {file_path}: {e}")
            return False

    def clean_temp_files(self) -> int:
        """Clean up temporary files."""
        temp_patterns = [
            '*.pyc',
            '*.pyo',
            '__pycache__',
            '*.log',
            '*.tmp',
            '.pytest_cache',
            '.coverage',
            'htmlcov',
            '.mypy_cache',
            '.ruff_cache'
        ]

        cleaned_files = 0

        for pattern in temp_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    file_path.unlink()
                    cleaned_files += 1
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
                    cleaned_files += 1

        return cleaned_files

    def run_black_formatting(self) -> bool:
        """Run black code formatter if available."""
        try:
            result = subprocess.run(['black', '--check', '--diff', str(self.project_root)],
                                  capture_output = True, text = True)
            if result.returncode != 0:
                print("Running black formatter...")
                subprocess.run(['black', str(self.project_root)], check = True)
                return True
            return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Black formatter not available, skipping...")
            return False

    def run_isort(self) -> bool:
        """Run isort import sorter if available."""
        try:
            result = subprocess.run(['isort', '--check - only', '--diff', str(self.project_root)],
                                  capture_output = True, text = True)
            if result.returncode != 0:
                print("Running isort...")
                subprocess.run(['isort', str(self.project_root)], check = True)
                return True
            return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("isort not available, skipping...")
            return False

    def run_ruff_linting(self) -> bool:
        """Run ruff linter if available."""
        try:
            result = subprocess.run(['ruff', 'check', '--fix', str(self.project_root)],
                                  capture_output = True, text = True)
            if result.returncode != 0:
                print("Running ruff linter...")
                return True
            return False
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ruff not available, skipping...")
            return False

    def clean_file(self, file_path: Path) -> bool:
        """
        Clean a single Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            True if any changes were made
        """
        changes_made = False

        # Remove unused imports
        if self.remove_unused_imports(file_path):
            changes_made = True

        # Fix formatting
        if self.fix_code_formatting(file_path):
            changes_made = True

        # Remove dead code
        if self.remove_dead_code(file_path):
            changes_made = True

        if changes_made:
            self.cleanup_stats["files_cleaned"] += 1

        return changes_made

    def run_cleanup(self) -> Dict[str, Any]:
        """
        Run comprehensive code cleanup.

        Returns:
            Dictionary with cleanup statistics
        """
        print("Starting code cleanup...")

        # Find Python files
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files")

        # Clean each file
        for file_path in python_files:
            self.cleanup_stats["files_processed"] += 1
            if self.clean_file(file_path):
                print(f"Cleaned: {file_path}")

        # Run external tools
        print("\nRunning external tools...")

        # Black formatting
        if self.run_black_formatting():
            print("Applied black formatting")

        # isort
        if self.run_isort():
            print("Applied isort import sorting")

        # ruff
        if self.run_ruff_linting():
            print("Applied ruff linting fixes")

        # Clean temporary files
        temp_files = self.clean_temp_files()
        if temp_files > 0:
            print(f"Cleaned {temp_files} temporary files")

        print(f"\nCleanup complete!")
        print(f"Files processed: {self.cleanup_stats['files_processed']}")
        print(f"Files cleaned: {self.cleanup_stats['files_cleaned']}")
        print(f"Unused imports removed: {self.cleanup_stats['unused_imports_removed']}")
        print(f"Formatting fixes: {self.cleanup_stats['formatting_fixes']}")
        print(f"Dead code removed: {self.cleanup_stats['dead_code_removed']}")

        return self.cleanup_stats

def main():
    """Main function."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."

    cleanup = CodeCleanup(project_root)
    stats = cleanup.run_cleanup()

    return 0 if stats["files_cleaned"] > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
