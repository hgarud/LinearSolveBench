"""The benchmark's two explicit numerical families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    selector: str
    id: str
    manifest: str
    accuracy_contract: str


NS_MESH_PDE = Family(
    selector="ns_mesh_pde",
    id="ns-mesh-pde",
    manifest="ns_mesh_pde/dev.json",
    accuracy_contract="ns-mesh-accuracy-v1",
)

FLASH = Family(
    selector="flash",
    id="magnetic_diffusion_flash",
    manifest="flash/dev.json",
    accuracy_contract="flash-replay-accuracy-v1",
)

_FAMILIES = {family.selector: family for family in (NS_MESH_PDE, FLASH)}


def resolve_family(selector: str) -> Family:
    """Return one of the two benchmark families by its public selector."""
    try:
        return _FAMILIES[selector]
    except KeyError as exc:
        raise ValueError(
            f"unsupported family {selector!r}; choose ns_mesh_pde or flash"
        ) from exc
