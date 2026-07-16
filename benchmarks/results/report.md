# Rextio-torch Phase A product-route benchmark

Primary rows call the generated Rextio wrapper in two persistent processes; only `REXTIO_NATIVE_MODE` differs. Context lanes are not Rextio claims.

## Expansion GO gate

- Verdict: **NO-GO**
- Aggregate geomean fallback/native speedup: `1.098517311212796`
- Aggregate speedup 95% CI: `[1.0915072173110354, 1.1055328595091287]`
- Target cells with native/fallback CI upper < 1.0: `4`

Blocking reasons:
- `aggregate-geomean-speedup-below-1.2:1.098517`
- `aggregate-bootstrap-ci-low-below-1.1:1.091507`

## Cells

| Case | Role | Batch | In | Out | Native ms | Fallback ms | Median n/f | Speedup f/n | n/f CI95 | Eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| t1 | target | 1 | 32 | 32 | 0.0031 | 0.0035 | 0.8849862389404508 | 1.1277642184078405 | [0.8725536155209724, 0.9015917858998861] | True |
| t2 | target | 1 | 128 | 64 | 0.0042 | 0.0047 | 0.8961465392393713 | 1.1098426041352583 | [0.8936608693221381, 0.9087685487385442] | True |
| t3 | target | 8 | 64 | 64 | 0.0050 | 0.0054 | 0.9212927099209668 | 1.0964442181556182 | [0.9009957915418491, 0.9216209949772967] | True |
| t4 | target | 16 | 128 | 64 | 0.0056 | 0.0059 | 0.9356084555244741 | 1.0611123657767931 | [0.9312154721772532, 0.9570529604898226] | True |
| c1 | context | 64 | 128 | 64 | 0.0070 | 0.0074 | 0.950794729210012 | 1.0480535644700835 | [0.9336523814115598, 0.9738745167538017] | True |
| c2 | context | 256 | 256 | 128 | 0.0238 | 0.0243 | 0.9813502942772467 | 1.0209133712546146 | [0.9606637367332719, 0.9978610642736034] | True |

## Explicit non-claims

- Results apply only to this Mac / recorded machine; no cross-machine claim.
- Timed scope is boundary-inclusive product latency through the generated Rextio wrapper (not internal-only native kernel timing).
- No CUDA, MPS, or training/autograd performance claim.
- No extrapolation beyond the six frozen cells and recorded pins.
- Direct eager and torch.compile are context-only; they never rescue the Rextio expansion GO gate.
- Compilation and first-call warm-up are excluded from steady-state samples.
- Smoke runs are not performance evidence.
