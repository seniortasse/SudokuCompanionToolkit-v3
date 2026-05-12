from __future__ import annotations


def main() -> int:
    try:
        from python.publishing.print_specs import list_print_format_specs
    except Exception as exc:
        print(f"ERROR importing print format registry: {exc}", flush=True)
        return 1

    print("=" * 72, flush=True)
    print("Available print formats", flush=True)
    print("=" * 72, flush=True)

    for spec in list_print_format_specs():
        print(f"- {spec.format_id}", flush=True)
        print(f"    vendor:        {spec.vendor}", flush=True)
        print(f"    binding_type:  {spec.binding_type}", flush=True)
        print(f"    trim:          {spec.trim_width_in} x {spec.trim_height_in} in", flush=True)
        print(f"    bleed:         {spec.bleed_in} in", flush=True)
        print(f"    paper_options: {', '.join(spec.paper_options) if spec.paper_options else '-'}", flush=True)
        print(f"    color_options: {', '.join(spec.color_options) if spec.color_options else '-'}", flush=True)
        print(f"    description:   {spec.description or '-'}", flush=True)
        print("", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())