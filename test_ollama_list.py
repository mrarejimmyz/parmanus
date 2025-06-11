#!/usr/bin/env python3
"""
Simple test to debug the Ollama client list method response
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama


def test_ollama_list():
    try:
        client = ollama.Client(host="http://localhost:11434")
        models = client.list()
        print(f"Type of models: {type(models)}")
        print(f"Models response: {models}")

        if hasattr(models, "models"):
            print(f"models.models: {models.models}")
            if models.models:
                print(f"First model type: {type(models.models[0])}")
                print(f"First model: {models.models[0]}")
                print(f"First model attributes: {dir(models.models[0])}")

        return models
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    print("🔍 Testing Ollama client list method...")
    test_ollama_list()
