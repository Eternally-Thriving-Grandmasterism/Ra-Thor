# Safe Python snippet (stdlib only)

```python
import os
import json

def load_local_config(path: str) -> dict:
    """Read a local JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    cfg = load_local_config("config.json")
    print("loaded keys:", list(cfg.keys()))
```

Does not spawn processes, does not call eval, no network, no remote loaders.
