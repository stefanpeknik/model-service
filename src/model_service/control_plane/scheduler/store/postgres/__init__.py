"""Executions in PostgreSQL: PostgresStorage (storage.py) reads and writes rows.

Row locks guard each record's changes, one advisory lock serializes claims, and
database time governs leases. The rules live in the shared ExecutionRepository.
"""
