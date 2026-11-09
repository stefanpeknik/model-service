"""Durable execution state rules, without storage or a worker.

transitions.ExecutionTransitions owns every state change; domain.py and
contracts.py hold the records and safe failure types it works on.
"""
