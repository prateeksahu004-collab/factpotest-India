# Model card: FactProtest credibility pipeline (v2)

## Intended use
Help readers, journalists, researchers and officials judge how well-supported a citizen's opinion about an Indian
protest is, by checking its factual claims against an encyclopedic source and rating its reasoning and civility.
It is a decision aid: it does not decide which side of a protest is right, and it should not be used to remove,
rank or penalise people.

## Components

| Stage | Model / method | Output |
|---|---|---|
| Injection screening | `meta-llama/llama-prompt-guard-2-86m` (Groq) | probability the text tries to manipulate the system; ≥ 0.5 caps the score at 20 |
| Policy moderation (live only) | `openai/gpt-oss-safeguard-20b` with a written policy (incitement, hate, harassment, doxxing) | violation flag + category |
| Claim extraction + rubric | `openai/gpt-oss-20b`, JSON mode, prompt `analyze/v3` (states each protest's demand so stance is measured against it) | ≤ 3 atomic factual claims; 1–5 ratings for reasoning, specificity, civility, emotional intensity; predicted stance |
| Retrieval | BM25 + `BAAI/bge-small-en-v1.5` (ONNX), Reciprocal Rank Fusion (k=60), top-3 passages within the protest's KB slice | evidence passages |
| Verification | `openai/gpt-oss-120b`, JSON mode, prompt `verify/v2`, evidence-only instruction | SUPPORTED / REFUTED / NOT_ENOUGH_INFO + confidence + cited passage ids |
| Score | `scoring.credibility`: 0.4·evidence + 0.4·reasoning + 0.2·civility (renormalised when there are no claims) | 0–100 + flags |
| Stance classifier | logistic regression on TF-IDF word+char n-grams or bge embeddings (best by leave-one-protest-out macro-F1) | Support / Oppose / Neutral probabilities |
| Coordination detector | cosine ≥ 0.88 between posts on the same protest within 180 min, clusters of ≥ 3 | campaign alerts |

## Evaluation
See `evals/README.md` for the protocol and `evals/results/summary.json` (also rendered on the app's
**Model Card & Evals** page) for numbers: retrieval recall/MRR (including paraphrased queries), verification
accuracy and macro-F1 per configuration (including oracle evidence), end-to-end misinformation recall and false-alarm
rate, score ROC-AUC, stance macro-F1 under leave-one-protest-out CV, campaign-detection precision/recall, and
run-to-run stability.

## Fairness
The score should depend on evidence and reasoning, not on which side an opinion takes. `analytics.score_bias_audit`
compares score distributions across declared stances (Kruskal–Wallis, Mann–Whitney, rank-biserial r) on the corpus
and live on the Data Insights page. Writing style is a second risk: short, informal or Hinglish comments tend to
receive lower reasoning and specificity ratings, so the score partly measures articulateness. Users should read the
per-component breakdown rather than the single number.

## Known failure modes
* Claims about events after the pinned Wikipedia revisions, or details Wikipedia omits, become NOT_ENOUGH_INFO.
* The extractor sometimes turns value judgements into "claims" (e.g. "the repeal was beneficial"); the verifier
  usually returns NOT_ENOUGH_INFO for these.
* Sarcasm and mixed-language text reduce stance accuracy for both the classifier and the LLM.
* LLM outputs vary between runs; see the consistency results.

## Data
Wikipedia (CC BY-SA 4.0) for evidence; a synthetic, clearly labelled opinion corpus for development and evaluation.
No personal data is collected beyond the name a user chooses to display.
