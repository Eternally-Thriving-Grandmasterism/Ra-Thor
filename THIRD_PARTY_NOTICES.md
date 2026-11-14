# Third-party notices

Files under `js/vendor/`, the files under `third-party/`, and the third-party AI models and model code that Lattice Chat downloads or runs are not part of the Ra-Thor Work. They are licensed by their own authors under their own terms. They are not licensed by Autonomicity Games Inc., and AG-SML and COMMERCIAL_LICENSE.md do not apply to them. Public page: https://rathor.ai/third-party-models.html

## Models (downloaded by the user's browser from Hugging Face, mlc-ai quantized builds; not stored in this repository)

- Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct: Llama 3.2 Community License Agreement and Llama 3.2 Acceptable Use Policy (`third-party/llama-3.2/`). Built with Llama.
  Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.
- gemma-2-2b-it (not currently offered in Lattice Chat): Gemma Terms of Use and Gemma Prohibited Use Policy (`third-party/gemma/`).
  Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms
  The gemma-2-2b-it files previously offered in Lattice Chat were quantized (modified) by MLC (mlc-ai) from google/gemma-2-2b-it.
- Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct: Apache License 2.0 (`third-party/qwen2.5/LICENSE.txt`).
- SmolLM2-360M-Instruct: Apache License 2.0 (`third-party/apache-2.0/LICENSE.txt`).
- Phi-3.5-mini-instruct: MIT License, Copyright (c) Microsoft Corporation (`third-party/phi-3.5/`).

## Software

- WebLLM 0.2.85 (`js/vendor/web-llm/0.2.85/`): Apache License 2.0, see `js/vendor/web-llm/0.2.85/LICENSE`. Unmodified copy of the npm package `@mlc-ai/web-llm@0.2.85` `lib/index.js`.
- Apache TVM (runtime included in WebLLM):
  Apache TVM
  Copyright 2019-2023 The Apache Software Foundation
  This product includes software developed at The Apache Software Foundation (http://www.apache.org/).
- Model code files (.wasm): fetched by the user's browser from https://github.com/mlc-ai/binary-mlc-llm-libs. That repository carries no licence statement from its publisher. Not stored in this repository.
