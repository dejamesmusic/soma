# the arithmetic of a shared model

james blight · 2026-11-25 · version 2 · note · verdict: open

## current evidence note

this version has no structured protocol review yet; consult its methods.


## original summary

what it costs to run one soma model for many people. a 1b model on one rtx pro 6000 holds 22 lanes for about $1,964 a month on secure cloud, around the clock. if six to ten people share a lane, that is 132 to 220 seats at about $9 to $15 each. use is training: every lane learns from what its seat reads and says. one model's lanes live on one gpu, so a model holds a community of about two hundred, and grows by sister models, not bigger machines. the training to make a 1b fluent is a few hundred dollars; running it is the cost.

published: 2026-11-25 20:48:41.276697+00:00

evidence: runpod list prices read nov 25 · lanes measured on an h100 and an rtx pro 6000 · training speed measured on an h100 · seats per lane assumed, not measured

[latest version](/api/developments/the-arithmetic-of-a-shared-model.md)

prices are runpod's list prices of 2026-11-25, in us dollars per hour. speeds are measured; seats per lane are assumed.

## the short version

a model's cost is gpu hours. a gpu's hours are shared by its lanes, and a lane is shared by the people taking turns on
it. everything else rounds to zero.

## training

the training harness runs a 1b at 28–31k tokens a second on one h100.

| tokens | time | community ($2.69/h) | secure ($4.39/h) |
|---|---|---:|---:|
| 10 billion | 4 days | about $255 | about $415 |
| 20 billion | 8 days | about $510 | about $830 |

## running

| gpu | a month, around the clock | lanes, 1b | a lane, a month |
|---|---:|---:|---:|
| h100 sxm, secure | $3,205 | 24 | $134 |
| rtx pro 6000, secure | $1,964 | 22 | $89 |
| rtx pro 6000, community | $1,234 | 22 | $56 |

- **seats per lane: assumed six to ten.** nobody talks without pause. that makes 132–220 seats on one pro 6000, at
  about $9–15 each a month on secure cloud.
- **storage is small.** a conversation's store is about 0.8 GB, about $0.06 a month. the model's volume, a few dollars.
- **startup**: 2 min 48 s to 4 min 10 s from renting to ready, mostly the provider placing and starting the machine.
  warm hours remove it.

## the ceiling

one model's lanes share its weights on one gpu. a second gpu would be a second copy, and the two would drift as each
learned. so one model holds a community of about two hundred. beyond that: sister models, forks that grow their own way.

## use is training

every lane learns from what its seat reads and says. twenty-four lanes reading sixteen hours a day would be about
27 billion tokens a month, more than pretraining. the gpu that serves is the gpu that trains.

## what's honestly unknown

- **seats per lane.** the assumption that sets everything above. it needs real use.
- **throughput in production.** the lanes split one stream's speed in production: reading through the product costs
  about $146 per billion tokens, against $25–40 in the harness.
- **many teachers.** learning is on for everyone a model is shared with; its owner can turn it off person by person.
  what two hundred teachers do to one set of weights is not measured.


