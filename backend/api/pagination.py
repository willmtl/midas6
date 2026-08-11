"""Cursor (keyset) pagination for the dashboard list endpoints — powers infinite scroll.

Why cursor, not offset: the study tables are large (StockStudy ~23k rows) and append/recompute
while being read; offset pagination skips/duplicates rows when the underlying set shifts. Cursor
pagination keys off the ordering value itself, so scrolling stays stable during a recompute.

Trade-off with the client-side sortable tables: a cursor page only holds ~50 rows, so sorting can
no longer happen fully client-side. Instead the frontend sends ?ordering=<col> and we re-issue the
query server-side from a fresh cursor. Each view exposes a small whitelist of sortable columns
(ORDERING maps a public column name -> a concrete DB field) so a client can't order by an arbitrary
or nullable field (CursorPagination requires a non-null, total-ish ordering key).
"""
from rest_framework.pagination import CursorPagination
from rest_framework.response import Response


class DashboardCursorPagination(CursorPagination):
    page_size = 50
    page_size_query_param = "page_size"
    max_page_size = 200
    # default ordering; each view passes its own via paginate(...)
    ordering = "-id"

    def paginate(self, queryset, request, *, ordering, last_updated=None, extra=None):
        """Paginate `queryset` ordered by `ordering` (str or tuple of DB fields, '-' = desc).
        Stashes `last_updated` (ISO str or None) and any `extra` top-level keys to merge into the
        response envelope. Returns the page (list of rows) — call get_paginated_response(page)."""
        self.ordering = ordering
        self._last_updated = last_updated
        self._extra = extra or {}
        return self.paginate_queryset(queryset, request, view=None)

    def get_paginated_response(self, data):
        payload = {
            "results": data,
            "next": self.get_next_link(),      # opaque cursor URL, or None at the end
            "previous": self.get_previous_link(),
            "last_updated": self._last_updated,
        }
        payload.update(self._extra)
        return Response(payload)


def resolve_ordering(request, whitelist, default, default_dir="desc"):
    """Map ?ordering=<key>[&dir=asc|desc] to a concrete DB ordering tuple.
    `whitelist` maps public column key -> DB field name (no leading '-'). Falls back to `default`
    (a DB field name) with `default_dir` when the key isn't whitelisted / absent. Always appends
    'id' as a unique tiebreaker so the cursor is a total order even when the metric has ties."""
    key = request.query_params.get("ordering")
    field = whitelist.get(key)
    if field is None:
        field, direction = default, default_dir
    else:
        direction = request.query_params.get("dir", "desc")
    prefix = "-" if direction != "asc" else ""
    tie = "id" if direction == "asc" else "-id"
    return (f"{prefix}{field}", tie)
