FROM pgvector/pgvector:pg15

# Copy initialization scripts
# These run on first startup to initialize the DB
COPY schema.sql /docker-entrypoint-initdb.d/10-schema.sql
