"""Use verified request timing and text relations when model tokens are unavailable."""
import json
from .logical import load_joined_calls, read_serving_config


def load_audited_logical(cell):
    summary_path = cell/'cleaning_summary.json'
    if not summary_path.exists():
        return None
    audit = json.loads(summary_path.read_text())
    if audit.get('physical_token_prefix_reconstructed') is not False:
        return None
    verified = json.loads((cell/'per_request.json').read_text())
    calls = load_joined_calls(cell)
    assert [c['run_id'] for c in calls] == [c['provider_request_id'] for c in verified]
    for c in calls:
        c.update(logical=0, logical_aligned=0, role_logical=0, our_tokens=0,
                 fresh_input=c['input'], logical_source_candidates=[],
                 realized_source_candidates=[], exact_hash_source_indices=[],
                 elapsed_s=(c['start_ms']-calls[0]['start_ms'])/1000)
    result = dict(cell=cell.name, n_calls=len(calls), per_call=calls,
                  tokenizer='unavailable', token_source={'mode': 'verified text only'},
                  our_input_tokens=None, openai_input_tokens=audit['input_tokens'],
                  logical_frac=None, logical_aligned_frac=None, logical_128_frac=None,
                  gap_frac=None, realized_frac=None, cacheread_tokens=None,
                  logical_tokens=None, logical_aligned_tokens=None,
                  block_demand={}, candidate_count_table=[], temporal_metrics={},
                  serving_config=read_serving_config(cell), cache_geometry={},
                  runtime={'vendors': sorted({c['vendor'] for c in calls}),
                           'models': sorted({c['model'] for c in calls})},
                  text_lineage=verified, token_reuse_unmeasured=True)
    return result
