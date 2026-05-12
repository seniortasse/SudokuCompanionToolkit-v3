__all__ = [
    "export_book_pdf",
    "export_book_interior_pdf",
    "export_book_cover_pdf",
    "LayoutProfile",
    "get_layout_profile",
    "BuiltBookRenderModel",
    "PublicationRenderContext",
    "load_built_book_render_model",
    "load_publication_render_context",
]


def __getattr__(name: str):
    if name == "export_book_pdf":
        from .book_pdf_exporter import export_book_pdf
        return export_book_pdf
    if name == "export_book_interior_pdf":
        from .interior_pdf_exporter import export_book_interior_pdf
        return export_book_interior_pdf
    if name == "export_book_cover_pdf":
        from .cover_pdf_exporter import export_book_cover_pdf
        return export_book_cover_pdf
    if name in {"LayoutProfile", "get_layout_profile"}:
        from .layout_profiles import LayoutProfile, get_layout_profile
        return {"LayoutProfile": LayoutProfile, "get_layout_profile": get_layout_profile}[name]
    if name in {
        "BuiltBookRenderModel",
        "PublicationRenderContext",
        "load_built_book_render_model",
        "load_publication_render_context",
    }:
        from .render_models import (
            BuiltBookRenderModel,
            PublicationRenderContext,
            load_built_book_render_model,
            load_publication_render_context,
        )
        return {
            "BuiltBookRenderModel": BuiltBookRenderModel,
            "PublicationRenderContext": PublicationRenderContext,
            "load_built_book_render_model": load_built_book_render_model,
            "load_publication_render_context": load_publication_render_context,
        }[name]
    raise AttributeError(name)