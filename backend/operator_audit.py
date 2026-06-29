"""No-op stand-in for the portal's operator audit (splatlab is single-operator)."""
async def audit_operator_event(*args, **kwargs) -> None:
    return None
