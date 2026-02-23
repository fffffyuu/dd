from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlantMetadata:
    scientific_name: str
    common_name: str
    description: str
    watering: str
    sunlight: str
    soil: str
    temperature: str


PLANT_CATALOG: dict[str, PlantMetadata] = {
    "fiddle_leaf_fig": PlantMetadata(
        scientific_name="Ficus lyrata",
        common_name="Fiddle Leaf Fig",
        description="A broadleaf tropical ornamental popular for indoor decor.",
        watering="Water when top 2-3 cm of soil is dry.",
        sunlight="Bright indirect light, 6+ hours daily.",
        soil="Well-draining potting mix with perlite.",
        temperature="18-27°C",
    ),
    "snake_plant": PlantMetadata(
        scientific_name="Dracaena trifasciata",
        common_name="Snake Plant",
        description="Hardy succulent-like foliage plant tolerant to low light.",
        watering="Every 2-3 weeks; let soil fully dry.",
        sunlight="Low to bright indirect light.",
        soil="Sandy fast-draining soil.",
        temperature="15-30°C",
    ),
    "money_plant": PlantMetadata(
        scientific_name="Epipremnum aureum",
        common_name="Money Plant",
        description="Trailing vine with variegated leaves, easy for beginners.",
        watering="Once weekly or when topsoil dries.",
        sunlight="Medium to bright indirect sunlight.",
        soil="Loamy mix with coco coir and compost.",
        temperature="17-29°C",
    ),
}


def get_metadata(plant_key: str) -> PlantMetadata:
    return PLANT_CATALOG.get(plant_key, PLANT_CATALOG["money_plant"])
