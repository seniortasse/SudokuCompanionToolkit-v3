from .assignment_ledger import (
    ensure_inventory_entry,
    get_inventory_entry,
    is_record_assigned_in_library,
    list_assigned_record_ids,
    register_assignment,
    sync_catalog_statuses_from_inventory,
    unregister_assignments_for_book,
)
from .assignment_rules import filter_records_available_for_library
from .library_inventory_store import load_library_inventory, save_library_inventory


def can_remove_record_from_catalog(*args, **kwargs):
    from .removal_guard import can_remove_record_from_catalog as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "ensure_inventory_entry",
    "get_inventory_entry",
    "is_record_assigned_in_library",
    "list_assigned_record_ids",
    "register_assignment",
    "sync_catalog_statuses_from_inventory",
    "unregister_assignments_for_book",
    "filter_records_available_for_library",
    "load_library_inventory",
    "save_library_inventory",
    "can_remove_record_from_catalog",
]