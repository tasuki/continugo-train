"""Experimental ContinuGo policy/value network; game rules remain in Rust."""

from .model import NetworkConfig, PolicyValueNet, resolution_for_diameter

__all__ = ["NetworkConfig", "PolicyValueNet", "resolution_for_diameter"]
