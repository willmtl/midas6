"""Limit/offset pagination for the dashboard list endpoints — powers infinite scroll.

Why offset, not cursor: these tables are sorted by a user-chosen column, and many of those columns are
nullable (t_stat, avg_mae, clean_pct, peak_avg, ret_90d, …). Cursor/keyset pagination needs a non-null,
near-unique ordering key, so it cannot back arbitrary-column sort — the very thing the sortable-column
UX requires. Offset pagination orders by any column (NULLS LAST) at any depth; the tables are small
enough (<=~26k rows) that even a deep offset stays fast. Each view passes an ORDERING whitelist so a
client can't order by an un-indexed / arbitrary field.

Response envelope (what the frontend `usePagedList` hook consumes):
    {results: [...], next_offset: <int|null>, total: <int>, last_updated: <iso|null>, ...extra}
`next_offset` is the offset to request for the following page, or null at the end.
"""
from django.db.models import F
from rest_framework.response import Response

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


def resolve_ordering(request, whitelist, default, default_dir="desc"):
    """Map ?ordering=<key>[&dir=asc|desc] to a list of order_by args (NULLS LAST) with an `id`
    tiebreaker so the slice is a total order even when the sort column has ties.
    `whitelist` maps a public column key -> a concrete DB field name (no leading '-')."""
    key = request.query_params.get("ordering")
    field = whitelist.get(key)
    if field is None:
        field, direction = default, default_dir
    else:
        direction = request.query_params.get("dir", "desc")
    asc = direction == "asc"
    primary = F(field).asc(nulls_last=True) if asc else F(field).desc(nulls_last=True)
    tie = F("id").asc() if asc else F("id").desc()
    return [primary, tie]


def _read_int(request, name, default):
    try:
        return int(request.query_params.get(name, default))
    except (TypeError, ValueError):
        return default


def paginate_offset(request, queryset, ordering):
    """Order `queryset` by `ordering` (a list of order_by args) and slice by ?offset=&page_size=.
    Returns (page, next_offset, total): `page` is a list (of model instances or .values() dicts,
    matching the queryset), `next_offset` is the offset for the next page or None at the end."""
    limit = max(1, min(_read_int(request, "page_size", DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE))
    offset = max(0, _read_int(request, "offset", 0))
    total = queryset.count()
    page = list(queryset.order_by(*ordering)[offset:offset + limit])
    next_offset = offset + limit if (offset + limit) < total else None
    return page, next_offset, total


def paged_response(rows, next_offset, total, *, last_updated=None, extra=None):
    """Build the standard infinite-scroll envelope."""
    payload = {"results": rows, "next_offset": next_offset, "total": total,
               "last_updated": last_updated}
    if extra:
        payload.update(extra)
    return Response(payload)
