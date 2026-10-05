# Business Entity Resolution: Methodology

> Team: Neural Nexus · Final submission: **v4 Run A + cross-encoder**, public leaderboard **0.978** (macro F0.5).
> All numbers come from the full-data run (`work/v4/reports/cut_report.txt`, `final_report.txt`,
> `test_diagnostics.txt`); [est] marks an estimate.

## 1. Overview

For every Source 1 (S1) entity we return the Source 2 / Source 3 records that describe the same
business. The pipeline is a cascade: high-recall retrieval, two re-ranking stages that shrink the
candidate list, a pair classifier, a cross-encoder on the records the classifier is unsure about,
and a decision rule tuned for F0.5.

```
TSV files -> multi-view normalization -> per country: 3 sparse channels + bounded exact blocks
          -> stage-1 candidate ranker (up to 100 per record)
          -> new sources: ranked big-key blocks, despaced / loose-romanization keys and channels
          -> stage-2 re-ranker (XGBoost pairwise ranking) -> top 5 + score floor, <= 25 per record
             + address rescue
          -> 118 pair features -> 3-fold grouped XGBoost -> cross-encoder on contested records
          -> exclusive assignment + threshold rule tuned on held-out entities at test density
          -> matching_results.tsv, candidate_pairs.tsv
```

Everything is built from the provided training data only. No external database, API or
geocoder is used. The cross-encoder is used: its starting weights are
`intfloat/multilingual-e5-small` (MIT, 118M parameters), fine-tuned on training pairs only;
nothing is looked up at prediction time.

## 2. What the data told us

Full-data statistics from our exploratory pass over all seven files:

| fact | consequence |
|---|---|
| Train: 2.21M S1, 5.03M S2, 5.29M S3 (US, India). Test: 1.73M S1, 4.89M S2, 5.08M S3 (US, India, France) | work per country, in chunks, with integer ids |
| No true pair crosses countries | country is a free, exact blocking key |
| Every S2/S3 record matches at most one S1 (strict 1:N) | decide per record: keep only its best S1 |
| 5.6% of S1 entities are singletons; ~26% of S2/S3 records match nothing, many of them near-copies of a real business (e.g. one door-number digit changed) | an empty prediction must be a real option; door-number features |
| Only 42% of true pairs share a sorted, suffix-free name; 15.8% share no name token | name similarity alone is not enough; addresses and transliteration matter |
| 13.9% of true pairs have a non-ASCII S2/S3 name (Indic scripts, accents); S2 is 5.4% Devanagari alone | romanize Devanagari, Bengali, Tamil, Telugu, ...; fold accents |
| ZIP agrees in 92.7% and state in 98.6% of true pairs when both present; Telangana/Andhra Pradesh are mixed | 3-state agreement flags; merge those two states |
| 38% of S1 names are shared by several entities ('primary care group' x253, 'bordeaux club sarl' x205) | the address must break ties; name commonness is a feature |
| Frequent test S2/S3 names are 'cc', 'sc', 'ac', 'pc', 'lc' | acronym noise ('Lille Club' -> 'LC') needs an acronym feature |
| S1 writes 'Unit' in 8.6% of addresses, S2 in 0.4% (as '#'); S2 is 28% all caps, 4% domain-style names, 3.4% empty addresses | normalization rules below |
| Test has ~5.8 S2/S3 records per S1 vs 4.7 in train | validate at test density (section 6) |
| France (test only): SARL/SAS/EURL/SA/SCI suffixes, 'Rue' in 66% of addresses | legal forms and French address words in the same country-agnostic rules |

## 3. Normalization

Each field keeps several views instead of one aggressively cleaned string (`normalize.py`,
`views4.py`), computed once per distinct string, in parallel, identically for every source and
country:

* Names: `name_raw_clean` (NFKC, original script), `name_full` (romanized: Indic scripts run by
  run with indic-transliteration, accents folded; legal forms, numbers, word order and DBA
  kept), `name_core` (honorifics removed), `name_base` (legal forms removed, the comparison
  view), `name_tokens_sorted`, `name_nospace`, a phonetic key per word, a loose romanization
  that maps everyday Indian spellings and ITRANS output to one key (`laxmi` / `lakshmi`,
  `praivet` / `private`, `teknoloji` / `technology`), the DBA name, the canonical legal form, an
  acronym, script and domain-style flags. Generic words are never removed.
* Addresses: `address_full` (unit words kept as `unit` / `fl`, padding zeros removed from house
  numbers), address tokens and phonetic keys, house / unit / floor number, postcode as written
  (leading zeros kept, e.g. `06000`) and its prefix, state, numeric tokens, missing and
  non-Latin flags. Compound door numbers (`8-2-293/82`, `3/4`, `G-4`) stay one token.

## 4. Candidate generation (blocking)

Per country partition (aliases such as `USA` / `IND` merged; blank-country records search every
S1), each S2/S3 record retrieves its top 40 S1 (plus ties) in three TF-IDF channels (sublinear
TF, IDF from S1, cosine): **name** (character 3-grams of `name_full` + phonetic word keys),
**address** (words, numbers, their phonetic keys, joined neighbours) and **joint** (name and
address tokens, prefixed per side). Records with no searchable address get a name top-120.

A **soft document-frequency cap** replaces max_df: a feature in more than 2% of the partition's
S1 records stays in every vector and every cosine but is left out of the retrieval posting
lists, and every record still retrieves with its two rarest common features. No record becomes
unsearchable because its name is common.

**Bounded rescue blocks** add exact-key candidates, each key used only when at most 100 S1
share it: `name_full`, sorted name tokens, sorted phonetic name, `address_full`, postcode +
house number, postcode + a rare name token, state + house number + a rare address token, DBA
to name.

**Candidate ranker (stage 1).** The union is scored by XGBoost (80 depth-5 trees) on cheap
features: the three cosines with their ranks and gaps inside the record, source flags, token
overlaps, postcode / state / house / unit / name agreement, name frequency, missingness. It is
fitted on sampled training records, never on records of tune or holdout entities. A record keeps
its 5 best candidates plus every candidate above a score floor, up to 100. There is no cap per
S1 entity.

**Stage 2 (no re-blocking).** On top of the stored candidates, new sources add pairs: ranked
big-key blocks (a name key shared by 100 to 5,000 S1s is no longer skipped: the record keeps
the 20 S1s under that key with the best address cosine), exact despaced-name and
loose-romanization keys, and two more sparse channels (character n-grams of the loose
romanization and of the despaced name). A stage-2 XGBoost ranker (`rank:pairwise`, grouped by
record, up to 400 depth-6 trees, 73 features, fitted on 150k training records only) orders the
expanded list of ~105 candidates per record. Each record keeps its 5 best, then every candidate
above a floor (-4.37, chosen on held-out training records to keep 99.8% of their true pairs),
up to 25, plus an address rescue. Held-out training records keep 9.2 candidates per record;
test keeps ~10.2 [est].

**Recall** (true pairs of tune + holdout entities, never seen by either ranker):

| layer | pair recall | oracle ceiling |
|---|---|---|
| stage-1 union (all channels and blocks) | 0.99324 | 0.99788 |
| stage-1 stored candidates (<= 100 per record) | 0.98932 | 0.99656 |
| stage-2 expanded set | 0.99187 | 0.99747 |
| **kept after the stage-2 cut (scored by the pair model)** | **0.98937** | **0.99671** |

The oracle ceiling is the macro F0.5 of a perfect classifier on the candidates. Kept recall by
segment: India 0.9856, US 0.9940, zero name-token overlap 0.9784, non-Latin name 0.9763, common
name 0.9735, missing address 0.8996.

## 5. Model and features

**Features** (`features.py`, `features4.py`, 118 in total): the ranker features and the stage-1
and stage-2 scores, ranks and gaps inside the record; string comparisons on the normalized
views: edit similarity of the raw-script and romanized names, token-sort and token-set
similarity of `name_base`, phonetic-key similarity, Jaro-Winkler of the space-free name, exact
full / core / sorted / despaced / loose name keys, best DBA similarity, acronym match, length
difference, name comparisons without legal words wherever they sit, domain / handle
containment, first / last name-token agreement, share of unmatched name tokens; address
token-set / token-sort similarity with and without state and country; postcode, state, house,
unit and floor agreement; door-number comparisons (equal, prefix, one-digit substitution,
truncation, minimum edit distance). No feature counts records per S1 or depends on the number of
S2/S3 records per S1 (test has about 24% more than train).

**Pair model.** One XGBoost binary model (depth 8, learning rate 0.08, subsample and column
sample 0.8, up to 1,500 trees with early stopping on an inner split) in 3 folds grouped by S1
entity: 12.73M candidate pairs of 300,000 training entities, excluding records of tune / holdout
entities and the stress look-alikes. Out-of-fold log loss 0.0075. Test pairs are scored with the
mean of the fold models. Strongest features by gain: stage-2 score, stage-1 score, stage-2 rank,
door-number edit distance, stage-2 gap, door-number truncation, loose-name key.

**Cross-encoder.** `multilingual-e5-small` with a two-class head, reading "name | address" of
both sides in the original script (at most 96 tokens). Fine-tuned for 1 epoch (AdamW, learning
rate 3e-5, batch 64) on 600k pairs of training entities: all true pairs, and false pairs mixed
1:3 between hard (stage-2 rank <= 2) and ordinary ones. It scores only the top three S1 of each
contested record (best pair-model probability above 0.05 and either below 0.95 or within 0.2 of
the runner-up), about 3.6M test pairs [est] instead of ~100M. A logistic stacker (pair-model
logit, cross-encoder logit, their product, rank-1 / rank-2 flags), fitted on tune records,
re-chooses each contested record's S1. It was fine-tuned and scored on a Kaggle T4 GPU
(`kaggle_cross.py`); everything else runs on CPU.

**Decision.** (1) Each S2/S3 record keeps only its best S1 (strict 1:N). (2) Candidate rules
are scored on the **tune** entities at test density: global thresholds 0.20-0.875, and per-S1
expected-F0.5 rules (exact Poisson-binomial, empty prefix allowed) with and without isotonic
calibration. (3) The best rule is frozen before the **holdout** entities are scored. Chosen:
**threshold p >= 0.75** (expected-F u=0.05 calibrated was close: holdout 0.98388).

Model size and licence: XGBoost (Apache-2.0): three pair-model boosters of at most 1,500 depth-8
trees, an 80-tree stage-1 ranker and a 400-tree stage-2 ranker; one 118M-parameter encoder
(MIT). All far below the 8B limit.

## 6. Validation

Each train country is sampled down to its test S1 count so every record meets as many
look-alike competitors as in test, keeping all records of each sampled entity and unmatched
records at the same rate. S1 entities are split into disjoint roles: train (models), tune
(stacker, calibration and rule, fitted at test density), holdout (40,000 entities, reported
score). A **density-stress** evaluation adds real look-alike records whose entity is not in the
index up to test's S2/S3-per-S1 density.

| metric | pair model only | + cross-encoder (final) |
|---|---|---|
| holdout macro F0.5, standard density | 0.98245 | 0.98541 |
| **holdout macro F0.5, test density (stress)** | 0.98102 | **0.98414** |
| India / US at test density | 0.97948 / 0.98291 | 0.98373 / 0.98465 |
| holdout candidate ceiling / model gap | 0.99658 / 0.0156 | 0.99658 / 0.0124 |
| best threshold vs expected-F rule on holdout | 0.98102 vs 0.98042 | 0.98414 vs 0.98388 |
| public leaderboard | 0.975 | **0.978** |

Holdout loss at test density (final model): S1 entities missing some true records 0.0090,
a wrong match 0.0034, predicted empty 0.0024, a false match on a no-match S1 0.0011. The
cross-encoder mostly recovers true records the pair model left below the threshold (India
+0.0043). The stress holdout has tracked the leaderboard to within ~0.006 across versions
(v1 0.959 / 0.942, v3 0.964 / 0.954, v4 0.981 / 0.975, v4 + cross 0.984 / 0.978).

## 7. France and other unseen countries

France has no training data. Nothing in the pipeline uses the country label except to form
partitions; the normalization handles French legal forms, elided articles, `cedex` and
leading-zero postcodes. Because France cannot be validated with labels,
`test_diagnostics.txt` compares each test country with the training reference. France is in
range: 5.5 records per S1 (India 5.8, US 5.8), 62.5% of records matched (training reference at
test density 58-65%), 94.8% of S1 non-empty, 3.6 matches per non-empty S1 (reference 3.5-3.6),
median best probability 0.999. An EM estimate puts the share of assigned records that are true
at 0.64 for France and 0.59 overall (0.63 on tune). France has more candidates per record (13.0
vs 9.7 India, 7.7 US) and slightly less similar matched addresses (median token-set 0.967 vs
1.0). `audit_france.tsv` holds 100 random France entities with their predicted and rejected
records for a manual precision check.

## 8. Engineering and reproducibility

* Hardware: one AWS r8i.2xlarge (8 vCPU, 64 GB RAM, no GPU) plus a Kaggle T4 for the
  cross-encoder. Stage-1 blocking took 6.4 h once and is cached; the v4 run took 4.7 h end to
  end; the cross-encoder import and re-prediction take minutes (test pair scores are cached).
* Stages cache to disk with a fingerprint of their settings and code, chained across stages;
  blocking and feature passes resume from the last finished chunk.
* Worker processes are spawned, pin every native thread pool to one thread and memory-map the
  indexes and strings; the process count follows the container's CPU quota.
* Fixed random seeds; `README.md` gives the commands and pinned versions. A self-check mirrors
  the submission rules; the official validator is run as well.
* Exact commands of the submitted run (this package adds an optional no-address setting that
  was not used; `--s2-keep-noaddr 0` reproduces the submitted run):

```bash
python src/run.py --data $D --work $W --s2-keep-noaddr 0 --out output_v4
python src/run.py --data $D --work $W --s2-keep-noaddr 0 --stage cross --cross kaggle   # export
# Kaggle T4: python src/kaggle_cross.py --export kaggle_export --out .
C="--data $D --work $W --s2-keep-noaddr 0 --cross kaggle --cross-scores cross_scores.parquet"
python src/run.py $C --stage cross && python src/run.py $C --stage evaluate
python src/run.py $C --stage predict --out output_v4_cross    # submitted files
```

From scratch (a new work folder, only the provided train/test data and this folder), one
command regenerates both files: `bash code/business_entity_resolution/reproduce.sh DATA_DIR`
on a GPU machine, or `CROSS=kaggle bash ... reproduce.sh DATA_DIR` on a CPU machine with the
Kaggle round trip (see the README section "Regenerating the submission from scratch"). The
cross-encoder is fine-tuned again from the public `intfloat/multilingual-e5-small` weights, so
its scores, and the final files, can differ slightly from the submitted ones.
