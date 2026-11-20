# backends/ — third_party, vendored agents & weight placeholders

This is **not** a second workflow layer.

| Path | Role |
|------|------|
| `checkpoints/` | **Weights not in git.** Put `Pi3X_model.safetensors` + `moge-2-vitl/model.pt` here, or use sibling `../pi3_moge_lerobot/checkpoints` via `EGO_NAV_FORGE_PI3_ROOT`. |
| `third_party/` | Vendored Pi3 / MoGe / utils3d **source** (no weight files). |
| `indoor_twin_agent/` | Vendored **indoor_twin_agent v37** (code + `assets/person_supplied.*` + `blender_lib`). Used by finalize module `IndoorTwinScene`. |

Pose modules may also load pipeline code from sibling `../pi3_moge_lerobot` (`EGO_NAV_FORGE_PI3_ROOT`).  
Scene twin agent override: `EGO_NAV_FORGE_TWIN_ROOT`.

`src/ego_nav_forge/modules/*` are the **only** user-facing building blocks.
