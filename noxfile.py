from __future__ import annotations

import nox
import nox_uv

nox.options.default_venv_backend = "uv"


@nox_uv.session(uv_groups=["test"])
def tests(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    # session.install(".[test]")
    session.run("pytest", *session.posargs)


@nox_uv.session(uv_groups=["docs"])
def docs(session: nox.Session) -> None:
    """
    Build the docs. Pass "serve" to serve.
    """

    # session.install("uv", ".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    if session.posargs:
        if "serve" in session.posargs:
            print("Launching docs at http://localhost:8000/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", "8000", "-d", "_build/html")
        else:
            print("Unsupported argument to docs")


@nox_uv.session(uv_groups=["dev"])
def benchmark(session: nox.Session) -> None:
    """
    Run runtime benchmarks.
    """

    session.run("python", "runtime_benchmark.py", *session.posargs)
