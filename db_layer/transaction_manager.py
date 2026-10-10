from contextlib import contextmanager
from typing import Generator
from sqlalchemy.orm import Session
from db_layer.database import SessionLocal

@contextmanager
def transaction() -> Generator[Session, None, None]:
    db = SessionLocal()       # 1. creează session nou
    try:
        yield db              # 2. execută codul din with-block
        db.commit()           # 3. totul OK → scrie în DB
    except Exception:
        db.rollback()         # 4. excepție → anulează tranzacția
        raise
    finally:
        db.close()            # 5. mereu — eliberează conexiunea
