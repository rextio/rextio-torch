# Rextio Torch Phase B benchmark

- Verdict: **NO-GO**
- Plugin SHA: `8d72a7ea7ba8c28bd798df7a6495fde6264c03df`
- Core SHA: `4ba053d1ff28978ccf69f9543e0124595bdebbc0`

## Dual comparison gates

| Comparator | Verdict | Aggregate speedup | 95% CI | CI cells |
|---|---|---:|---:|---:|
| fallback | NO-GO | 1.1510459729432514 | [1.144691189915756, 1.1577218191395733] | 4 |
| direct_eager | NO-GO | 1.134274863261328 | [1.1275315260928205, 1.1410234869930402] | 4 |

## Blocking reasons

- fallback:aggregate-geomean-speedup-below-1.2:1.151046
- direct_eager:aggregate-geomean-speedup-below-1.2:1.134275

## Non-claims

- No CUDA, MPS, training, autograd, or optimizer claim.
- No claim beyond the exact six frozen deep alternating MLP cells.
- torch.compile is context-only and unavailable compilation cannot rescue or fail a comparison.
- A GO verdict would not authorize publication, tagging, visibility changes, or PyPI upload.
