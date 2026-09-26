"""Shared setup: the tests never reach Jev or the local model, so the key only has to exist for `flow.jev` to build its agent at import."""

import os

os.environ["TYPESAFE_API_KEY"] = "test-key-never-sent"
