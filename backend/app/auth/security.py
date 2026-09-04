"""
Password hashing utilities using bcrypt.
"""
import bcrypt


def hash_password(plain_password: str) -> str:
    """Hash a password using bcrypt."""
    pwd_bytes = plain_password.encode("utf-8")
    # Truncate to 72 bytes if needed (standard bcrypt maximum length)
    pwd_bytes = pwd_bytes[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a bcrypt hash."""
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception:
        return False
