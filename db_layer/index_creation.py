import logging
from sqlalchemy import text
from database import engine

logger = logging.getLogger(__name__)

INDEX_NAME = "ix_chunks_embedding_ivfflat"
TABLE = "document_chunks"
COLUMN = "embedding"


def create_index() -> None:
    with engine.begin() as conn:
        # 1. Drop dacă există (idempotent)
        conn.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))

        # 2. Build IVFFLAT cu cosine — în prod: CREATE INDEX CONCURRENTLY (nu blochează tabela)
        conn.execute(text(f"""
        
            CREATE INDEX {INDEX_NAME}
            ON {TABLE}
            USING ivfflat ({COLUMN} vector_cosine_ops)
            WITH (lists = 100)
        """))

        # 3. Verificare: index există și dimensiunea lui
        result = conn.execute(text("""
            SELECT pg_size_pretty(pg_relation_size(:idx))
        """), {"idx": INDEX_NAME}).scalar()

        logger.info("Index %s created, size=%s", INDEX_NAME, result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    create_index()
