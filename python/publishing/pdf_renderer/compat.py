from __future__ import annotations


def patch_hashlib_usedforsecurity() -> None:
    """
    Make hashlib, lower-level OpenSSL-backed constructors, and ReportLab's
    cached md5 helpers tolerant of the usedforsecurity=... keyword argument.

    Some library stacks call:
        hashlib.md5(..., usedforsecurity=False)

    On certain Python/OpenSSL builds, especially some Python 3.8 Windows setups,
    this raises:
        TypeError: 'usedforsecurity' is an invalid keyword argument for openssl_md5()

    Important:
    ReportLab may cache md5 helpers during import, so patching hashlib.md5 alone
    is not always enough after ReportLab modules have already been imported.
    This function is therefore safe to call repeatedly. The hashlib-level patch
    is applied once, but the ReportLab module-level md5 aliases are refreshed
    every time.
    """
    import hashlib
    import importlib

    def _wrap(func):
        def _patched(*args, **kwargs):
            kwargs.pop("usedforsecurity", None)
            return func(*args, **kwargs)

        return _patched

    if not getattr(hashlib, "_sudoku_usedforsecurity_patch_applied", False):
        hashlib_names = [
            "md5",
            "sha1",
            "sha224",
            "sha256",
            "sha384",
            "sha512",
            "blake2b",
            "blake2s",
            "sha3_224",
            "sha3_256",
            "sha3_384",
            "sha3_512",
        ]

        for name in hashlib_names:
            if hasattr(hashlib, name):
                original = getattr(hashlib, name)
                setattr(hashlib, name, _wrap(original))

        if hasattr(hashlib, "new"):
            original_new = hashlib.new

            def _patched_new(name, data=b"", **kwargs):
                kwargs.pop("usedforsecurity", None)
                return original_new(name, data, **kwargs)

            hashlib.new = _patched_new

        # Patch lower-level OpenSSL-backed constructors too, because some stacks
        # reach them directly and bypass hashlib.md5 / hashlib.new wrappers.
        try:
            import _hashlib  # type: ignore
        except Exception:
            _hashlib = None  # type: ignore

        if _hashlib is not None:
            low_level_names = [
                "openssl_md5",
                "openssl_sha1",
                "openssl_sha224",
                "openssl_sha256",
                "openssl_sha384",
                "openssl_sha512",
            ]
            for name in low_level_names:
                if hasattr(_hashlib, name):
                    original = getattr(_hashlib, name)
                    setattr(_hashlib, name, _wrap(original))

        hashlib._sudoku_usedforsecurity_patch_applied = True

    def _md5_compat(data=b"", *args, **kwargs):
        kwargs.pop("usedforsecurity", None)
        return hashlib.md5(data, *args, **kwargs)

    # ReportLab can cache md5 at module import time. Refresh these aliases every
    # time this function is called, even if the hashlib-level patch was already
    # applied earlier.
    for module_name in (
        "reportlab.lib.utils",
        "reportlab.pdfbase.pdfdoc",
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        if hasattr(module, "md5"):
            try:
                setattr(module, "md5", _md5_compat)
            except Exception:
                pass