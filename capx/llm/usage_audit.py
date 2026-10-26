"""Audit provider accounting without rewriting it or inferring actual billing."""

TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'total_tokens')


def complete_sum(values):
    return sum(values) if all(isinstance(x, int) and not isinstance(x, bool) for x in values) else None


def audit_usage(usage, *, protocol=None, output_limit=None):
    reported = {key:usage.get(key) for key in TOKEN_FIELDS}
    anomalies = []
    if (all(isinstance(x, int) for x in reported.values())
            and reported['input_tokens']+reported['output_tokens'] != reported['total_tokens']):
        anomalies.append('reported_total_tokens_mismatch')
    if (protocol == 'responses' and isinstance(output_limit, int)
            and isinstance(reported['output_tokens'], int) and reported['output_tokens'] > output_limit):
        anomalies.append('reported_output_tokens_exceed_requested_limit')
    reasoning = usage.get('reasoning_tokens')
    if (isinstance(reasoning, int) and isinstance(reported['output_tokens'], int)
            and reasoning > reported['output_tokens']):
        anomalies.append('reported_reasoning_exceeds_output')
    raw = usage.get('provider_usage') or {}
    attribution = raw.get('attribution') if isinstance(raw, dict) else None
    details = None
    ratios = {}
    if isinstance(attribution, dict):
        # Item-level totals already include nested content[]; never count both.
        # request_fields (e.g. instructions) are separate from items.
        groups = [attribution.get(name, {}) for name in ('items', 'request_fields')]
        parts = [part for group in groups if isinstance(group, dict)
                 for part in group.values() if isinstance(part, dict)]
        if parts:
            details = {key:complete_sum([part.get(key) for part in parts])
                       for key in ('input_tokens', 'output_tokens')}
            details['total_tokens'] = complete_sum(list(details.values()))
            for key in ('input_tokens', 'output_tokens'):
                a, b = reported[key], details[key]
                if isinstance(a, int) and isinstance(b, int):
                    if a != b:
                        anomalies.append('provider_attribution_' + key + '_difference')
                    ratios[key] = a/b if b else None
    complete = all(isinstance(x, int) for x in reported.values())
    status = ('needs_review' if anomalies else
              'provider_reported_unverified' if complete else 'unavailable')
    return {'status':status, 'reported_totals':reported, 'attribution_totals':details,
            'attribution_ratios':ratios, 'anomalies':anomalies,
            'comparable_totals':reported if complete and not anomalies else dict.fromkeys(TOKEN_FIELDS),
            'independently_verified':False,
            'note':'Attribution is a provider extension; completeness and billing are unverified. '
                   'Differences may include unitemized usage. No scaling or replacement is applied.'}


def summarize_accounting(events):
    audits = [audit_usage(e.get('usage', {}), protocol=e.get('protocol'),
              output_limit=e.get('request', {}).get('max_output_tokens_requested')) for e in events]
    comparable = {key:complete_sum([a['comparable_totals'][key] for a in audits]) for key in TOKEN_FIELDS}
    known = {key:sum(v for a in audits if isinstance((v:=a['reported_totals'][key]), int))
             if any(isinstance(a['reported_totals'][key], int) for a in audits) else (0 if not events else None)
             for key in TOKEN_FIELDS}
    detail = {key:complete_sum([(a['attribution_totals'] or {}).get(key) for a in audits])
              for key in TOKEN_FIELDS}
    reviewed = sum(a['status']=='needs_review' for a in audits)
    unavailable = sum(a['status']=='unavailable' for a in audits)
    return {'status':'needs_review' if reviewed else 'unavailable' if unavailable else
                    'no_requests' if not events else 'provider_reported_unverified',
            'needs_review_query_count':reviewed, 'unavailable_query_count':unavailable,
            'comparable_totals':comparable, 'known_reported_totals':known,
            'attribution_totals':detail, 'independently_verified':False,
            'note':'Only internally consistent provider reports enter comparable totals; '
                   'these counts are not independently verified billing.'}
