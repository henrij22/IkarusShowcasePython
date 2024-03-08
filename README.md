# (Trimmed) IGA showcase using Ikarus' python bindings

This workspace provides an example for IGA analysis using [Ikarus](https://ikarus-project.github.io/dev/) with Python.

You can use the workspace with the provided devcontainer. Everything should be set up automatically for you.
The only dependency is Docker with WSL2 integration. It is also possible to run in a Github codespace.

### Troubleshooting

If Ikarus doesn't install / compile itself automatically you can run:

```bash
pip install --pre pyikarus --verbose --upgrade --no-build-isolation
```

If Python isn't able to find Ikarus, make sure to be in the correct python venv.
Run  `pip -V`. This should return something similar to

```bash
pip 23.3.1 from /dune/dune-common/build-cmake/dune-env/lib/python3.11/site-packages/pip (python 3.11)
```

If this is not the case enable the venv via

```bash
source /dune/dune-common/build-cmake/dune-env/bin/activate
```

LICENSE: If not otherwise stated: MIT
