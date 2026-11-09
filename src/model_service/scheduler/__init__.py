"""Durable execution state rules and the seams around them, without a worker.

transitions.ExecutionTransitions owns every state change; store.repository is the
persistence interface and providers is the provider interface. Persistence
implementations remain separate.
"""
