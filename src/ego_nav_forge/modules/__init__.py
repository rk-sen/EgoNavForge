"""Import modules so @register runs."""
from __future__ import annotations

from ego_nav_forge.modules import ingest as _ingest  # noqa: F401
from ego_nav_forge.modules import preprocess as _preprocess  # noqa: F401
from ego_nav_forge.modules import pose as _pose  # noqa: F401
from ego_nav_forge.modules import depth as _depth  # noqa: F401
from ego_nav_forge.modules import scale as _scale  # noqa: F401
from ego_nav_forge.modules import export as _export  # noqa: F401
from ego_nav_forge.modules import viz as _viz  # noqa: F401
from ego_nav_forge.modules import scene as _scene  # noqa: F401
