# InsightBot AI Deployment Guide

## Architecture Overview
InsightBot AI is deployed as a resilient service featuring:
- Asynchronous connection pooling via asyncpg.
- Strict Pydantic v2 schemas for all document models and REST contracts.
- Deterministic text chunking with configurable overlap.

## Database Configuration
The primary persistent database is PostgreSQL 17. The service connects to database `insightbot`.
Foreign-key cascading deletion ensures all chunk vectors are removed automatically when a parent document is deleted.
Content hash deduplication via SHA-256 prevents duplicate document storage.
