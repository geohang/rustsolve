"""Regenerate patches/gimli-<tag>-rustsolve.patch from apply_rustsolve.py.

usage: python make_patches.py <gimli git clone> [tag ...]   (default v1.5.0 v1.5.4 v1.5.5)

For each tag: export the tree (git archive), commit it to a scratch repository,
run apply_rustsolve.py, and write `git diff` as the patch (LF, no BOM).
"""
import io
import pathlib
import subprocess
import sys
import tarfile
import tempfile

HERE = pathlib.Path(__file__).resolve().parent


def git(cwd, *args, **kw):
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, check=True,
                          capture_output=True, **kw).stdout


def main():
    clone = pathlib.Path(sys.argv[1]).resolve()
    tags = sys.argv[2:] or ["v1.5.0", "v1.5.4", "v1.5.5"]
    for tag in tags:
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp) / "gimli"
            work.mkdir()
            tar = git(clone, "archive", "--format=tar", tag)
            with tarfile.open(fileobj=io.BytesIO(tar)) as t:
                t.extractall(work, filter="data")
            git(work, "init", "-q")
            git(work, "add", "-A")
            git(work, "-c", "user.name=x", "-c", "user.email=x@x", "commit", "-q", "-m", tag)
            subprocess.run([sys.executable, str(HERE / "apply_rustsolve.py"), str(work)], check=True)
            diff = git(work, "diff", "--no-color", "--no-ext-diff")
            out = HERE / "patches" / f"gimli-{tag}-rustsolve.patch"
            out.write_bytes(diff)
            print(f"{out.name}: {len(diff)} bytes")


if __name__ == "__main__":
    main()
