from typing import Optional, List
from sqlmodel import SQLModel


# 🔹 Entrada: Características de la propiedad
class PropertyCharacteristics(SQLModel):
    air_conditioning: Optional[bool] = None
    annex_garage: Optional[float] = None
    annex_storage: Optional[float] = None
    balcony: Optional[bool] = None
    building_status: Optional[float] = None
    built_in_wardrobes: Optional[bool] = None
    built_sqm: Optional[float] = None
    description: Optional[str] = None
    detached_garage: Optional[float] = None
    detached_storage: Optional[float] = None
    dressing_room: Optional[bool] = None
    energy_efficiency_certificate: Optional[float] = None
    entrance_hall: Optional[bool] = None
    exterior_wheelchair_access: Optional[bool] = None
    facing_direction: Optional[str] = None
    floor: Optional[float] = None
    fuel: Optional[float] = None
    green_areas: Optional[bool] = None
    heating: Optional[float] = None
    interior_wheelchair_access: Optional[bool] = None
    is_penthouse: Optional[bool] = None
    kitchen_layout: Optional[float] = None
    last_edited_at: Optional[str] = None
    laundry: Optional[bool] = None
    laundry_in_kitchen: Optional[bool] = None
    number_of_bathrooms: Optional[float] = None
    number_of_bedrooms: Optional[float] = None
    number_of_elevators: Optional[float] = None
    number_of_toilets: Optional[float] = None
    other_rooms: Optional[bool] = None
    pantry: Optional[bool] = None
    patio: Optional[bool] = None
    pool: Optional[bool] = None
    property_id: Optional[float] = None
    social_housing: Optional[bool] = None
    status: Optional[float] = None
    subtype: Optional[float] = None
    terrace: Optional[bool] = None
    title: Optional[str] = None
    type: Optional[float] = None
    usable_sqm: Optional[float] = None
    usage: Optional[float] = None
    year_of_construction: Optional[float] = None


# 🔹 Salida: Tamaño y distribución de espacios
class RoomSize(SQLModel):
    perimeter: float
    area: float


class PropertySpaces(SQLModel):
    kitchen: RoomSize
    livingRoom: RoomSize
    laundry: Optional[RoomSize] = None
    lounge: Optional[RoomSize] = None
    hall: Optional[RoomSize] = None
    dressing: Optional[RoomSize] = None
    corridors: Optional[RoomSize] = None
    other: Optional[RoomSize] = None
    bedrooms: List[RoomSize]
    bathrooms: List[RoomSize]
    toilets: List[RoomSize]
    nStairs: Optional[float] = None
    nSteps: Optional[float] = None
    nRooms: Optional[float] = None
