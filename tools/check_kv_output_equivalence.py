"""Compare recorded output token sequences without interpreting differences as errors."""
import argparse
import hashlib
import json
from pathlib import Path


def compare(reference, candidate):
    def read(rep):
        rows = [json.loads(line) for line in (rep/'requests.jsonl').read_text().splitlines() if line.strip()]
        keyed = {row['request_id']:row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError(f'Duplicate request IDs: {rep}')
        return keyed
    left, right = read(reference), read(candidate)
    if left.keys() != right.keys():
        raise ValueError('Request ID sets differ')
    records = []
    for rid, first in left.items():
        second = right[rid]
        if first['prompt_tokens'] != second['prompt_tokens']:
            raise ValueError(f'Prompt token counts differ: {rid}')
        a, b = first.get('output_token_ids'), second.get('output_token_ids')
        if not isinstance(a,list) or not isinstance(b,list):
            raise ValueError(f'Output token IDs were not recorded: {rid}')
        difference = next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)) if len(a)!=len(b) else None)
        records.append({'request_id':rid,'equal':a==b,'first_different_output_position':difference,'reference_cached_tokens':first.get('cached_tokens'),'candidate_cached_tokens':second.get('cached_tokens')})
    return {'reference':str(reference.resolve()),'candidate':str(candidate.resolve()),'requests':len(records),'identical_outputs':sum(row['equal'] for row in records),'different_outputs':sum(not row['equal'] for row in records),'source_sha256':{label:hashlib.sha256((path/'requests.jsonl').read_bytes()).hexdigest() for label,path in [('reference',reference),('candidate',candidate)]},'scope':'Exact output token equality for matched request IDs and prompt lengths. Verify original bundle identity separately. A difference does not establish its cause or which output is correct. Output positions are zero-based.','records':records}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.reference,args.candidate)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['requests','identical_outputs','different_outputs']}))
