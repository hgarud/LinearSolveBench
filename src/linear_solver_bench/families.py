"""The benchmark's two explicit numerical families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    selector: str
    id: str
    manifest: str
    accuracy_contract: str
    scoring_contract: str


NS_MESH_PDE = Family(
    selector="ns_mesh_pde",
    id="ns-mesh-pde",
    manifest="ns_mesh_pde/dev.json",
    accuracy_contract="ns-mesh-accuracy-v1",
    scoring_contract="dev-cases-solved-v1",
)

MAGNETIC_DIFFUSION_FLASH = Family(
    selector="magnetic_diffusion_flash",
    id="magnetic_diffusion_flash",
    manifest="magnetic_diffusion_flash/dev.json",
    accuracy_contract="flash-replay-accuracy-v1",
    scoring_contract="reference-speedup-v1",
)

_FAMILIES = {
    family.selector: family for family in (NS_MESH_PDE, MAGNETIC_DIFFUSION_FLASH)
}
FAMILY_SELECTORS = tuple(_FAMILIES)


def resolve_family(selector: str) -> Family:
    """Return one of the two benchmark families by its public selector."""
    try:
        return _FAMILIES[selector]
    except KeyError as exc:
        choices = ", ".join(FAMILY_SELECTORS)
        raise ValueError(
            f"unsupported family {selector!r}; choose from {choices}"
        ) from exc
