from __future__ import annotations

import hashlib
import hmac

ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

def access_code(sap_id: str, quiz_id: str, faculty_key: str) -> str:
    """Deterministic 8-character access code; SAP ID alone is insufficient."""
    message = f"{sap_id.strip().upper()}|{quiz_id.strip().upper()}".encode("utf-8")
    digest = hmac.new(faculty_key.encode("utf-8"), message, hashlib.sha256).digest()
    value = int.from_bytes(digest[:8], "big")
    chars=[]
    for _ in range(8):
        value, rem = divmod(value, len(ALPHABET))
        chars.append(ALPHABET[rem])
    return "".join(chars)

def verify_access_code(sap_id: str, quiz_id: str, supplied: str, faculty_key: str) -> bool:
    expected=access_code(sap_id, quiz_id, faculty_key)
    return hmac.compare_digest(expected, supplied.strip().upper())
