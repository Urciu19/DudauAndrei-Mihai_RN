from pathlib import Path
import hashlib

def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def hashes(folder: Path):
    m = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            m[p] = md5(p)
    return m

train = hashes(Path("data/train"))
val   = hashes(Path("data/val"))
test  = hashes(Path("data/test"))

train_hashes = set(train.values())
val_hashes   = set(val.values())
test_hashes  = set(test.values())

print("Train images:", len(train_hashes))
print("Val images:", len(val_hashes))
print("Test images:", len(test_hashes))

print("Overlap train-test:", len(train_hashes & test_hashes))
print("Overlap train-val:", len(train_hashes & val_hashes))
print("Overlap val-test:", len(val_hashes & test_hashes))

if len(train_hashes & test_hashes) > 0:
    print("\n[!] Data leakage: există imagini identice în train și test.")