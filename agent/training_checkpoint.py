"""Publish completed checkpoint directories without exposing partial saves."""
import os
import tempfile
from pathlib import Path

def publish_checkpoint(root, name, writer):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    final=root/name
    if final.exists(): raise FileExistsError(final)
    staging=Path(tempfile.mkdtemp(prefix='.partial-',dir=root))
    writer(staging)
    # A failed writer leaves a clearly marked partial directory, not a checkpoint.
    staging.rename(final)
    pointer=root.parent/'latest-checkpoint.txt'
    temporary=pointer.with_suffix('.pending')
    temporary.write_text(str(final.resolve()),encoding='utf-8')
    os.replace(temporary,pointer)
    return final
