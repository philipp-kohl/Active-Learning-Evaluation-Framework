def is_named_entity(label: str) -> bool:
    if label not in [None, "","0","O"]:
        return True
    return False