# soma-1b: a new frontier

james blight · 2026-11-19 · version 2 · result · verdict: alive

## current evidence note

editorial review 23 november 2026, for this specific notebook version. the original title and experiment text below are historical.

evaluation protocol: **multi_lane_training_stream**. one run; no matched transformer control; not a strictly source-ordered unseen-text evaluation; scaling equation and extrapolated floor withdrawn in version 2.

cost/accounting scope: reported run estimate; preparation, idle, storage and transfer costs are not separately itemised in the post; not an established all-in cost.

[structured evidence and available material](/api/evidence.json) · [developments](/developments)


## original summary

a 1.06B-parameter somatic core model trained on 1.012 billion FineWeb-Edu tokens using eight distant lanes sharing one learner, on one H100 in 27.5 hours for about $121. its final training-score window was 0.925 bits per byte. this is not a frozen held-out or strictly chronological evaluation; no matched 1B transformer control exists.

published: 2026-11-23 04:14:08.394910+00:00

evidence: one run, one seed; 1,011,515,392 scored training tokens of FineWeb-Edu; eight distant lanes share learner updates, each tick scored before its own update; prefix counting pass seeds each lane; not a strictly source-ordered prequential or frozen held-out result; one H100 SXM

[latest version](/api/developments/soma-1b-a-new-frontier.md)

## what we did

soma-1b is a somatic core model with a learner 2048 wide and 20 deep: 1,057,581,284 parameters, and an exact store of 0.82 GB beside them. the neural learner trained on 1,011,515,392 tokens of FineWeb-Edu, one training visit per token, split into eight distant lanes. the count stores also required a separate prefix-counting pass. each lane was scored before its current tick was learned from, but all lanes shared one learner: earlier updates from later stretches can already contain text later in source order. the scores below are training-stream measurements, not a strictly chronological unseen-text or frozen held-out evaluation.

it ran on one H100 for 27.5 hours, at about 12.9k tokens a second, for about $121 on runpod. eight lanes of 1024 tokens fed one learner; each lane's count store was seeded from the source prefix preceding that lane. gradients from all eight lanes were averaged before the shared update.

## how well it reads

bits per byte scored before each tick's update, averaged over processed-token windows across the lanes:

| tokens read | bits per byte |
|---|---|
| 100M | 1.176 |
| 250M | 1.055 |
| 500M | 0.982 |
| 750M | 0.943 |
| 900M | 0.931 |
| last 40M | 0.925 |

over the whole scored training run, it averaged 1.022. the exact store on its own reads at 1.586; the learner carries the rest.

the reported curve was still falling. version 1 gave a scaling equation whose stated units do not reproduce its reported scores or projections. that equation and the numerical extrapolations are withdrawn pending a reproducible fit; these data alone do not establish an asymptotic floor.

continued on a synthetic conversation set (conversation_worlds_v4), it reached 0.4 bits per byte. that set is very regular, so read this as how fast it takes up a new kind of text, not as a measure of english.

## what it remembers

a frozen windowed transformer's direct attention access ends at its window; learned weights or added memory may retain other information. somatic core combines attention with a count store and copy ring that can carry evidence beyond that window. at 370M tokens we gave the model an article to read online, then about 3,000 tokens of unrelated text, then measured how many bits it needed for names from the article. before it had read the article, and after:

| | before | after |
|---|---|---|
| Gauguin | 45.4 | 1.7 |
| Gérôme | 65.9 | 1.4 |
| Louisine Elder | 36.7 | 1.1 |
| Blohm & Voss | 89 | 18 |
| Paris (control) | 10.5 | 10.0 |

the names it met in the article became nearly free to predict, thousands of tokens after they left its 1024-token window. the control didn't move.

## against transformers

we haven't trained a transformer of the same size on the same billion tokens yet, so there's no matched comparison at this scale. the near-shape comparisons we have are smaller: on one gigabyte of FineWeb-Edu, a 768-wide somatic core scored 1.188 against 1.1925 for a transformer of the same shape, and a 1024-wide one scored 1.1646 against 1.178 for a 161M transformer (one seed each; 106M versus 92M parameters at width 768, and 178M versus 161M at width 1024). soma continued learning on the scored 50 MB; the transformers were trained offline and then frozen. the small lead is not established beyond run-to-run spread. a matched run at 1B remains open. [full controls](/api/developments/somatic-core-1gb.md).

## what it's like to talk to

it writes grammatical, fluent english. it drifts between topics and invents facts, as a base model that has read one billion tokens will, and it leans on what it read most recently. it is not an assistant. it's a model that keeps reading, and gets better as it does.

## what's next

more tokens, a matched transformer at the same size, and specialising it: reading deeply in one domain is where a memory like this should pay most.

published models can be found through [models](/models). access and available machines are reported by the live model page rather than guaranteed by this dated experiment.

## correction on 23 november 2026

version 2 distinguishes per-tick scoring from strictly chronological prequential evaluation, records the extra prefix-counting pass, and withdraws the inconsistent scaling equation and extrapolated floor. it also makes the smaller controls' parameter counts and evaluation regimes explicit. the reported measurements are unchanged. version 1 remains available at its versioned url. the lane protocol is implemented in `somatic_core/lanes.py` and the score windows in `tools/scale/pretrain.py`.


