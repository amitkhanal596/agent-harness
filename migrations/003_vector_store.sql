-- §3.3: retrieval over an embedded document corpus, stored in the same Postgres
-- instance that already holds runs, checkpoints, and events.
--
-- Using pgvector rather than a dedicated vector database is a deliberate choice: the
-- harness already depends on Postgres for durability, so retrieval adds no new
-- infrastructure, and a document's embedding lives transactionally beside its text.
--
-- `CREATE EXTENSION vector` requires pgvector to be installed on the server. The
-- migration runner tolerates its absence: retrieval is optional, and the rest of the
-- harness must still work without it.

DO $$
BEGIN
    CREATE EXTENSION IF NOT EXISTS vector;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pgvector unavailable (%); skipping vector store setup', SQLERRM;
END;
$$;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE TABLE IF NOT EXISTS documents (
            id          BIGSERIAL PRIMARY KEY,
            collection  TEXT        NOT NULL DEFAULT 'default',
            source      TEXT        NOT NULL,
            chunk_index INT         NOT NULL DEFAULT 0,
            content     TEXT        NOT NULL,
            -- 1024 dims matches voyage-3 / voyage-3-lite, the embedding models Anthropic
            -- recommends. Changing this requires re-embedding the corpus.
            embedding   vector(1024) NOT NULL,
            metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (collection, source, chunk_index)
        );

        -- HNSW beats IVFFlat for recall at these corpus sizes and needs no training
        -- step, so an index exists from the first insert rather than after a rebuild.
        CREATE INDEX IF NOT EXISTS idx_documents_embedding
            ON documents USING hnsw (embedding vector_cosine_ops);

        CREATE INDEX IF NOT EXISTS idx_documents_collection
            ON documents (collection);
    END IF;
END;
$$;
