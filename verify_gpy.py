import struct


def verify_gguf(filepath):
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic == b"GGUF":
                print(f"✅ {filepath} is a valid GGUF file")
                return True
            else:
                print(f"❌ {filepath} has invalid magic: {magic} (expected GGUF)")
                return False
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return False


verify_gguf("models/llava-model.gguf")
