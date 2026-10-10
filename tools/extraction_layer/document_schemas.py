from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    """Produs din factură."""
    denumire:    str
    cantitate:   int
    pret_unitar: float
    total:       float

class Invoice(BaseModel):
    """Schemă extracție date factură."""
    numar:  str
    data_emitarii:   str
    data_scadenta:  str
    client: str
    # Listă imbricată — LLM-ul populează automat structura
    produse: List[Product] = Field(default=[])
    total_general: float

class Contract(BaseModel):
    """Schemă extracție date contract."""

    numar:          str   = Field(description="Număr contract")
    data_incheiere: str   = Field(description="Data semnării")
    prestator:      str   = Field(description="Numele prestatorului")
    beneficiar:     str   = Field(description="Numele beneficiarului")
    valoare:        float = Field(description="Valoarea în RON")
    durata_luni:    int   = Field(description="Durata în luni")

    obligatii_prestator: List[str] = Field(
        default=[],
        description="Lista obligațiilor prestatorului",
    )