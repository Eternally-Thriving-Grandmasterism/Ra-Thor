# Clean Markdown with Code Fence

Here is a completely safe code example that only uses the standard library:

```python
import json
from pathlib import Path

def read_local_json(path: str):
    data = Path(path).read_text(encoding="utf-8")
    return json.loads(data)
```

The fence contains no eval, no exec, no subprocess, no network, and no remote loaders.
