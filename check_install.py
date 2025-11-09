#!/usr/bin/env python3
"""
Quick diagnostic script to verify installation and services.
"""

import sys
from pathlib import Path


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def check_python():
    """Check Python version."""
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    print("✅ Python version OK")
    return True


def check_dependencies():
    """Check Python dependencies."""
    print_header("Python Dependencies")

    required = [
        "click",
        "pydantic",
        "yaml",
        "fitz",
        "ollama",
        "numpy",
        "pandas",
        "datasketch",
        "requests",
        "tqdm",
        "sqlalchemy",
    ]

    missing = []
    for module in required:
        try:
            if module == "yaml":
                __import__("yaml")
            elif module == "fitz":
                __import__("fitz")
            else:
                __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing.append(module)

    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def check_ollama():
    """Check Ollama service."""
    print_header("Ollama Service")

    try:
        import ollama

        # Try to list models
        models_resp = ollama.list()
        print("✅ Ollama is running")

        # Normalize response into model names
        model_names = []
        items = None
        if hasattr(models_resp, "models"):
            items = getattr(models_resp, "models")
        elif isinstance(models_resp, dict):
            items = models_resp.get("models") or models_resp.get("data") or []
        elif isinstance(models_resp, list):
            items = models_resp

        if isinstance(items, list):
            for m in items:
                if isinstance(m, str):
                    model_names.append(m)
                elif isinstance(m, dict):
                    name = (
                        m.get("name")
                        or m.get("model")
                        or m.get("id")
                        or m.get("tag")
                        or ""
                    )
                    if name:
                        model_names.append(name)
                else:
                    # Typed object from ollama client (e.g., Model)
                    name = getattr(m, "name", None) or getattr(m, "model", None)
                    if isinstance(name, str) and name:
                        model_names.append(name)

        required_models = ["deepseek-r1:8b", "nomic-embed-text"]
        for model in required_models:
            if any((model == n) or n.startswith(model) or (model in n) for n in model_names):
                print(f"✅ {model} available")
            else:
                print(f"⚠️  {model} not found")
                print(f"   Run: ollama pull {model}")

        return True

    except Exception as e:
        print(f"❌ Ollama not running or model list unavailable: {e}")
        print("Install from: https://ollama.ai")
        return False


def check_grobid():
    """Check GROBID service."""
    print_header("GROBID Service")

    try:
        import requests

        response = requests.get("http://localhost:8070/api/isalive", timeout=5)

        if response.status_code == 200:
            print("✅ GROBID is running on port 8070")
            return True
        else:
            print(f"⚠️  GROBID returned status {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ GROBID not running: {e}")
        print("Start with: make grobid-start")
        print("Or: docker run -d -p 8070:8070 lfoppiano/grobid:0.8.0")
        return False


def check_tesseract():
    """Check Tesseract OCR."""
    print_header("Tesseract OCR")

    import subprocess

    try:
        result = subprocess.run(
            ["tesseract", "--version"], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            print(f"✅ {version_line}")
            return True
        else:
            print("❌ Tesseract check failed")
            return False

    except FileNotFoundError:
        print("❌ Tesseract not found")
        print("Install with: brew install tesseract")
        return False
    except Exception as e:
        print(f"⚠️  Error checking Tesseract: {e}")
        return False


def check_docker():
    """Check Docker."""
    print_header("Docker")

    import subprocess

    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
            return True
        else:
            print("❌ Docker check failed")
            return False

    except FileNotFoundError:
        print("❌ Docker not found")
        print("Install from: https://www.docker.com/products/docker-desktop")
        return False
    except Exception as e:
        print(f"⚠️  Error checking Docker: {e}")
        return False


def check_project_structure():
    """Check project files exist."""
    print_header("Project Structure")

    required_files = [
        "cli.py",
        "config.py",
        "requirements.txt",
        "core/inventory.py",
        "core/parser.py",
        "core/metadata.py",
        "core/manifest.py",
        "utils/hash.py",
        "cache/cache_manager.py",
    ]

    all_exist = True
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} missing")
            all_exist = False

    return all_exist


def main():
    """Run all checks."""
    print(
        """
    ╔═══════════════════════════════════════════╗
    ║   Research Assistant - Diagnostic Check  ║
    ╚═══════════════════════════════════════════╝
    """
    )

    checks = [
        ("Python", check_python),
        ("Dependencies", check_dependencies),
        ("Project Structure", check_project_structure),
        ("Docker", check_docker),
        ("Ollama", check_ollama),
        ("GROBID", check_grobid),
        ("Tesseract", check_tesseract),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            results[name] = False

    # Summary
    print_header("Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, status in results.items():
        symbol = "✅" if status else "❌"
        print(f"{symbol} {name}")

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\n🎉 All checks passed! You're ready to run the pipeline.")
        print("\nNext steps:")
        print("  1. python cli.py process --root-dir /path/to/papers --topic 'Your topic'")
        print("  2. Or: make run ROOT_DIR=/path/to/papers TOPIC='Your topic'")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("See TROUBLESHOOTING.md for help.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
