"""The two supported pilot tracks and their fixed scientific contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrackContract:
    family: str
    track: str
    accuracy_contract_id: str
    scoring_contract_id: str
    execution_contract_id: str = "independent-case-once-v1"
    repetitions: int = 1

    @property
    def contracts(self) -> dict[str, str]:
        return {
            "accuracy": self.accuracy_contract_id,
            "execution": self.execution_contract_id,
            "scoring": self.scoring_contract_id,
        }


TRACKS = (
    TrackContract(
        "ns-mesh-pde", "coverage", "ns-mesh-accuracy-v1", "ns-coverage-count-v1"
    ),
    TrackContract(
        "magnetic_diffusion_flash",
        "replay",
        "flash-replay-accuracy-v1",
        "flash-replay-speedup-v1",
    ),
)


def resolve_track(family: str, track: str) -> TrackContract:
    canonical = "ns-mesh-pde" if family == "ns_mesh_pde" else family
    for contract in TRACKS:
        if (contract.family, contract.track) == (canonical, track):
            return contract
    supported = ", ".join(f"{item.family}/{item.track}" for item in TRACKS)
    raise ValueError(f"unsupported family/track {family}/{track}; choose {supported}")


def family_listing() -> list[dict[str, object]]:
    return [asdict(contract) for contract in TRACKS]
