# Appendix

This appendix provides supplementary material for the paper, including experimental settings, full results, hard-sample analyses, and the annotation prompts used in the two-round labeling process.

> Note: In figure-heavy sections, image entries are provided as placeholders with paths and captions so the document can be reviewed and finalized more easily.

## Experimental hyperparameters

| Item | Setting |
|---|---|
| Data split | Train/Dev/Test = 80/10/10 (fixed) |
| Target label | `mm_label` |
| Max epochs | 10 |
| Early stopping | By *weighted F1* on Dev; patience = 2 |
| Learning rate | $2\times10^{-5}$ |
| Max length (text) | 256 |
| Batch size (train/eval) | 4 / 8 (default for trained models) |
| LLM inference | `temperature` = 0.0 |

*Table: Summary of experimental hyperparameters (condensed from the run configuration).*

## Full results table

### Text-only

| Model | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| RoBERTa-base | 0.6685 / 0.6627 / 0.7125 | 0.6712 / 0.6680 / 0.7160 | 0.6726 / 0.6660 / 0.7165 | 0.6726 / 0.6638 / 0.7130 |
| PhoBERT-base | **0.6848 / 0.6838 / 0.7518** | **0.6848 / 0.6838 / 0.7518** | 0.6821 / 0.6798 / 0.7301 | 0.6793 / 0.6776 / 0.7340 |
| mBERT | 0.6671 / 0.6650 / 0.7168 | 0.6386 / 0.6385 / 0.6970 | 0.6576 / 0.6353 / 0.7112 | 0.6399 / 0.6368 / 0.7015 |

### Image-only

| Model | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| ViT-B/32 | 0.5666 / 0.5666 / 0.5983 | -- | -- | -- |
| CLIP ViT-L/14 | **0.6603 / 0.6456 / 0.7101** | -- | -- | -- |

### Multimodal

| Model | s1 | s2 | s3 | s4 |
|---|---|---|---|---|
| DT4MID | 0.6929 / 0.6864 / 0.7449 | 0.6848 / 0.6841 / 0.7621 | 0.6807 / 0.6730 / 0.7476 | **0.6943 / 0.6884 / 0.7529** |
| CIRM | 0.6318 / 0.6292 / 0.6629 | 0.4375 / 0.3043 / 0.5336 | 0.4375 / 0.3043 / 0.5280 | 0.6128 / 0.5943 / 0.5772 |
| LLaVA-1.6-7B | 0.5639 / 0.3634 / -- | 0.5639 / 0.3634 / -- | 0.5639 / 0.3634 / -- | 0.5639 / 0.3634 / -- |
| Qwen3-VL-8B | 0.5734 / 0.5734 / -- | 0.5910 / 0.5908 / -- | 0.5734 / 0.5734 / -- | 0.5910 / 0.5908 / -- |
| ViMMSD [1] (staged gating) | 0.5815 / 0.3019 | 0.5761 / 0.2731 | 0.5666 / 0.2850 | 0.5829 / 0.3131 |
| ViMMSD [2] (hier. cross-attn.) | 0.6155 / 0.3043 | 0.6087 / 0.3053 | 0.6155 / 0.3020 | 0.6073 / 0.2881 |
| ViMMSD [3] (multimodal fusion) | 0.5095 / 0.3459 | 0.6182 / 0.4038 | 0.6087 / 0.4780 | 0.5883 / 0.4163 |

*Table: Full results on the* `test` *set (Accuracy / F1-macro / AUC).*

## Detailed multimodal results

| Model | Scenario | F1-macro | Accuracy | Precision | Recall | AUC |
|---|---|---:|---:|---:|---:|---:|
| DT4MID | s1 | 0.6864 | 0.6929 | 0.6917 | 0.6929 | 0.7449 |
| DT4MID | s2 | 0.6841 | 0.6848 | 0.6956 | 0.6848 | 0.7621 |
| DT4MID | s3 | 0.6730 | 0.6807 | 0.6790 | 0.6807 | 0.7476 |
| DT4MID | s4 | **0.6884** | **0.6943** | 0.6935 | **0.6943** | 0.7529 |
| ViMMSD [1] (staged gating) | s1 | 0.3019 | 0.5815 | 0.3784 | 0.5815 | -- |
| ViMMSD [1] (staged gating) | s2 | 0.2731 | 0.5761 | 0.3794 | 0.5761 | -- |
| ViMMSD [1] (staged gating) | s3 | 0.2850 | 0.5666 | 0.3624 | 0.5666 | -- |
| ViMMSD [1] (staged gating) | s4 | 0.3131 | 0.5829 | 0.3758 | 0.5829 | -- |
| ViMMSD [2] (hier. cross-attn.) | s1 | 0.3043 | 0.6155 | 0.4884 | 0.6155 | -- |
| ViMMSD [2] (hier. cross-attn.) | s2 | 0.3053 | 0.6087 | 0.4793 | 0.6087 | -- |
| ViMMSD [2] (hier. cross-attn.) | s3 | 0.3020 | 0.6155 | 0.4960 | 0.6155 | -- |
| ViMMSD [2] (hier. cross-attn.) | s4 | 0.2881 | 0.6073 | 0.4887 | 0.6073 | -- |
| ViMMSD [3] (multimodal fusion) | s1 | 0.3459 | 0.5095 | 0.4478 | 0.5095 | -- |
| ViMMSD [3] (multimodal fusion) | s2 | 0.4038 | 0.6182 | 0.5310 | 0.6182 | -- |
| ViMMSD [3] (multimodal fusion) | s3 | 0.4780 | 0.6087 | 0.5845 | 0.6087 | -- |
| ViMMSD [3] (multimodal fusion) | s4 | 0.4163 | 0.5883 | 0.5288 | 0.5883 | -- |

*Table: Detailed multimodal results on the test set for DT4MID and the three ViMMSD architectures across four scenarios.*

## Hard samples by scenario

This appendix visualizes *hard samples* (instances misclassified by many models simultaneously) for each *ablation* scenario. For each sample, we report (i) the gold label, (ii) the label combination $(T,I,M)$, and (iii) the list of models that made incorrect predictions under the corresponding scenario.

### s1 (no preprocessing, emoji kept)

**Figure**

- Image path: `figures/hard_samples/hs_id00595`
  - Suggested embed path: `![Figure](figures/hard_samples/hs_id00595)`
- Caption: s1 hard sample: false negative on a community/context-dependent case (ID 595).

**Figure**

- Image path: `figures/hard_samples/hs_id05990`
  - Suggested embed path: `![Figure](figures/hard_samples/hs_id05990)`
- Caption: s1 hard sample: false positive on a neutral background post (ID 5990).

**Figure**

- Image path: `figures/hard_samples/hs_id02861`
  - Suggested embed path: `![Figure](figures/hard_samples/hs_id02861)`
- Caption: s1 hard sample: the ``Tôi ...:'' pattern with a reaction image tends to trigger positive predictions (ID 2861).

### s2 (no preprocessing, emoji removed)

**Figure**

- Image path: `figures/hard_samples/hs_id05803`
  - Suggested embed path: `![Figure](figures/hard_samples/hs_id05803)`
- Caption: s2 hard sample: an everyday complaint/humor post misread as sarcasm by many models (ID 5803).

**Figure**

- Image path: `figures/hard_samples/hs_id05990`
  - Suggested embed path: `![Figure](figures/hard_samples/hs_id05990)`
- Caption: s2 hard sample: a recurrent false positive across multiple scenarios (ID 5990).

**Figure**

- Image path: `figures/hard_samples/hs_id04170`
  - Suggested embed path: `![Figure](figures/hard_samples/hs_id04170)`
- Caption: s2 hard sample: heavy laughter markers and text-rich images can be misleading, but the gold label is non-sarcastic (ID 4170).

### s3 (preprocessed, emoji kept)

Hard samples in s3 largely overlap with those in s2 on the *test* set (IDs 5803, 5990, 4170) and exhibit the same dominant FP pattern on $(0,0,0)$. We therefore omit repeated figures here for brevity.

### s4 (preprocessed, emoji removed)

**Figure**

- Image path: `figures/hard_samples/hs_id03587`
  - Suggested embed path: `![Figure](figures/hard_samples/hs_id03587)`
- Caption: s4 hard sample: requires contextual reasoning and text--image comparison to detect sarcasm (ID 3587).

**Figure**

- Image path: `figures/hard_samples/hs_id01602`
  - Suggested embed path: `![Figure](figures/hard_samples/hs_id01602)`
- Caption: s4 hard sample: sarcasm relies on community knowledge and evaluative stance, and is easily missed (ID 1602).

**Figure**

- Image path: `figures/hard_samples/hs_id01703`
  - Suggested embed path: `![Figure](figures/hard_samples/hs_id01703)`
- Caption: s4 hard sample: sarcasm/criticism depends on event context and text-rich screenshots (ID 1703).

## Prompts used in the two annotation rounds

### Round-1 prompt (binary label)

```text
You are an annotator for multimodal sarcasm detection in Vietnamese social media posts.

Your task is to determine whether a given post is non-sarcastic (label 0) or sarcastic (label 1),
by analyzing three modalities: text, images, and emoji. Not every input contains emoji.

Reason through all checks in order; do not skip any, even if the answer seems obvious.
Return exactly one valid JSON object and nothing else — no explanation, no markdown, no text outside the JSON.
All reasoning fields must be written in Vietnamese.
All JSON keys and enum values must be in English exactly as shown in the output schema.

Input:
[TEXT]
{text}

[IMAGES]
{images}

[OCR_TEXT]
{ocr_text}
Note: OCR_TEXT is automatically extracted from the image via OCR and may contain recognition errors (misspellings, missing diacritics, incorrect tokenization). Use OCR_TEXT only as auxiliary context to understand in-image text — do NOT treat it as primary evidence.

---

=== LABEL DEFINITIONS ===

Label 0 = Non-sarcastic: text, image, and emoji are consistent and aligned; a literal reading matches the author's intent.
Label 1 = Sarcastic: the author says/shows one thing but intends a different stance — cues can be subtle or require cultural/context knowledge. Typically falls into one of 7 types:
  1.1 Verbal: the text says the opposite of the intended meaning.
  1.2 Image-Text Conflict: the text and image directly contradict each other.
  1.3 Emoji-Text Conflict: emoji expresses an opposing emotion and flips the meaning of the text.
  1.4 Contextual: the text appears normal but sarcasm is only apparent with real-world context or common cultural templates.
  1.5 Self-deprecating: self-mockery via inversion or exaggeration.
  1.6 Hyperbolic: extreme exaggeration used to criticize implicitly.
  1.7 Multimodal: requires combining text + image + emoji + context to recognize sarcasm.
Label "INVALID": use when sarcasm cannot be determined because the input is faulty or insufficient. INVALID cases include:
  - caption and image are completely unrelated (two different topics with no semantic relation);
  - empty caption, only special characters, or not a coherent sentence;
  - missing image, or the image is too blurry/corrupted to recognize content;
  - caption language cannot be identified;
  - content is censored to the point that essential information is missing;
  - confidence < 0.4 and the caption–image relation cannot be determined.
  NOTE: do NOT assign INVALID merely because the content is hard to interpret, requires long thinking, or is politically sensitive.
        a slightly blurry image is NOT INVALID if content is still recognizable.
        if you can hypothesize a caption–image relation (even vaguely), choose the closest label and note uncertainty in validity.

---

=== REQUIRED ANALYSIS PROCEDURE ===

--- FIRST CHECK: VALIDITY ---
Is the caption semantically related to the image? Is the image readable and is the caption a coherent sentence?
→ If there is no semantic relation, or the input is faulty per INVALID cases above: stop, assign "INVALID", and clearly state the reason in validity.
→ If valid: continue.

--- SECOND CHECK: DIVERGENT THINKING ---
Form BOTH arguments with equal weight before deciding:
Non-Sarcastic case — argue that the post is entirely literal: modalities are aligned, the situation is mundane, and there are no sarcasm cues from the list above.
Sarcastic case — argue that the post is sarcastic: there is conflict, exaggeration, inversion, or any cue from the list above.

--- THIRD CHECK: EMOJI & SARCASM TYPE ---
- Does the emoji reinforce or invert the meaning?
- If the post is Label 1, which type (1.1–1.7) is most plausible?

--- FINAL VERDICT ---
Compare the two arguments and apply the tiebreaker rules:
  → If the Sarcastic case is stronger or tied AND the post uses sarcasm cues/formats common in Vietnamese.
  → If the Non-Sarcastic case clearly dominates and no cues from the list are present: assign label 0.
  → Do NOT use "no additional context" as a reason to automatically assign label 0 — contextual sarcasm (1.4) is valid even with only shared cultural knowledge.

---

=== OUTPUT FORMAT ===

Return EXACTLY this JSON. No text before or after.

{
  "llm_label": <0 | 1 | "INVALID">,
  "reasoning": {
    "non_sacarstic_case": "<the strongest argument that the post is literal (SECOND CHECK)>",
    "sacartic_case": "<the strongest argument that the post is sarcastic; if yes, specify type 1.x and concrete cues (SECOND CHECK)>",
    "emoji_reasoning": "<if emoji is present, explain whether it reinforces or inverts meaning>",
    "sacarsm_type": "<sarcasm type if label=1 per guideline; NULL otherwise>",
    "verdict": "<final decision rationale; apply tiebreaker if needed; mention whether evidence is Text_Only / ImageSet_Only (FINAL VERDICT)>"
  },
  "has_emoji": <0 | 1>,
  "needs_human_check": <"0 if the LLM is confident in the verdict; 1 if human verification is needed">
}

```

### Round-2 prompt (T/I/M labels)

```text
You are an annotator for Vietnamese multimodal sarcasm detection.

Your goal is to label the post carefully and conservatively.
Do not assume sarcasm unless there is clear evidence.
Return exactly one valid JSON object and nothing else.
Write reasoning values in Vietnamese. DO NOT write reasoning in English.

Important validity rule:
- Use "INVALID" only when the input is effectively unusable: the text is empty/meaningless or the image is missing/unreadable, so a reliable judgment is impossible.
- If there is enough usable evidence to judge, choose 0 or 1.

Input:
[TEXT]
{text}

[IMAGES]
{images}

[OCR_TEXT]
{ocr_text}

[ROUND1_BINARY_LABEL]
{label_round_1}

Use OCR only as supporting evidence because OCR may contain recognition errors.
Do not treat OCR as the main evidence if the image itself is readable.
The round-1 binary label is only prior context. You may disagree with it.
If your final verdict differs from the round-1 binary label, explain the reason more tightly and concretely in `reasoning.verdict`.

Important annotation convention
In this dataset, sarcasm is broader than strict "saying the opposite". It also includes common Vietnamese social-media mocking styles such as:
- cà khịa / đá xoáy / móc mỉa
- derisive rhetorical questions
- quoting others only to ridicule or dismiss them
- contemptuous or passive-aggressive dismissal
- mocking disbelief signaled by emoji or discourse markers like "ừ", "ha", "rồi đó" when clearly insincere

Do not assume direct criticism = non-sarcastic.
A post can still be sarcastic if the criticism is expressed as ridicule, sneering, mock imitation, rhetorical contempt, or social-media-style mockery, even without explicit literal reversal.

But do not overcall sarcasm.
Do NOT label sarcastic only because the post is funny, absurd, exaggerated, fandom-like, emotional, or emoji-heavy.
If the tone is merely playful, affectionate, surprised, or humorous without a clear mocking target or derisive stance, prefer 0.

Task
Evaluate the post in 3 steps:

Step 1 — Text-only (T)
Ignore the image. Read only TEXT.
Question: Would an ordinary Vietnamese reader detect sarcasm from the words alone?

Set T=1 only if there is clear textual sarcasm such as:
- verbal irony / saying the opposite of the intended meaning
- fake praise for an obviously bad situation
- strong hyperbole used critically
- mocking emoji that clearly flips the text meaning
- self-mocking contrast

Set T=0 if the text is literal, merely emotional, vague, or too ambiguous.

Step 2 — Image-only (I)
Ignore the text. Look only at the image.
Question: Would the image itself communicate sarcasm or irony without needing the caption?

Set I=1 only if the image alone provides clear evidence, for example:
- a known mocking meme / reaction template
- ironic text inside the image itself
- a visual contradiction inside the image itself
- an obviously failed situation framed as success inside the image

Set I=0 for ordinary images such as selfies, scenery, food, objects, luxury aesthetics, or serious photos without explicit irony.
Do not mark I=1 just because the image is funny, dramatic, weird, or aesthetically edited.

Step 3 — Multimodal final decision (MM)
Now consider the full post with BOTH text and image.
Question: When reading the whole post naturally, is the overall post sarcastic?

Set MM=1 when ANY of these is true:
- the text alone is already sarcastic and the image does not cancel that reading
- the image alone is already sarcastic and the text does not cancel that reading
- sarcasm mainly emerges from the relation between text and image

Typical MM=1 cases (can be more cases than the following list):
- praising text + clearly bad image
- humble/complaining text + obviously boastful image
- neutral text + clearly mocking meme image
- sarcastic text + neutral image

Set MM=0 if the whole post is best read literally, sincerely, or the evidence is too weak/speculative.

Important consistency rules:
- MM is a post-level decision, not only a contrast detector
- T=1 does NOT automatically force MM=1, but MM is usually 1 unless the image clearly changes/cancels the sarcastic reading
- I=1 does NOT automatically force MM=1, but MM can still be 1 even if T=0

Final label rule
- Output `final_label` using this exact rule:
  - final_label = 0 for (T,I,MM) in {(0,1,0), (1,0,0), (0,0,0)}
  - otherwise final_label = 1
- If the input is unusable under the strict validity rule above, output `final_label` = "INVALID"

Other fields
- has_emoji = 1 if TEXT contains emoji, else 0

Calibration examples
Use these examples to match the dataset convention:

Example 1 — sarcastic via derisive rhetorical question
- Text: "Chúng m ơi thật sự ảnh kỉ yếu nhất thiết chụp như thế à :))?"
- Image: awkward yearbook-style poses
- Why: This is not a sincere question. It is a mocking rhetorical question with a sneering tone.
- Output tendency: T=1, I=0, MM=1, final_label=1

Example 2 — sarcastic via contemptuous social-media mockery
- Text: a long rant attacking how fans keep defending a celebrity after repeated bad behavior, ending with a sneering smiley like ":)"
- Image: screenshot of fans expressing sympathy
- Why: Even though the criticism is direct, the tone is clearly cà khịa / contemptuous mockery, not just neutral criticism.
- Output tendency: T=1, I=0, MM=1, final_label=1

Example 3 — sarcastic via quoted ridicule / dismissive "ừ"
- Text: quote people defending someone ("xin lỗi cũng bị chửi...") and end with "ừ��‍♀️��‍♀️"
- Image: apology post screenshot
- Why: The quoted defense is being repeated to mock or dismiss it, not endorsed sincerely.
- Output tendency: T=1, I=0, MM=1, final_label=1

Example 4 — sarcastic via double-standard mock comparison
- Text: compare how men vs women are judged after saying something offensive
- Image: apology screenshot
- Why: The post frames a mocking, derisive comparison about social double standards rather than a neutral observation.
- Output tendency: T=1, I=0, MM=1, final_label=1

Example 5 — NOT sarcastic: playful / weird / absurd is not enough
- Text: "Có thể bạn chưa biết... 2 con mụ trong ảnh giờ đã cưới nhau��"
- Image: ordinary photo of two girls
- Why: The post is trollish / absurd / joking, but not clearly mocking a target with sarcastic stance.
- Output tendency: T=0, I=0, MM=0, final_label=0

Example 6 — NOT sarcastic: affectionate or excited tone with emoji
- Text: "gửi bé tiểu cường ở đây để thật điềm tĩnh khi xem, chứ các mom xịn thíaaaa ��"
- Image: cute bug/character image
- Why: Emoji, exaggeration, or cute/self-joking tone alone do not make it sarcastic.
- Output tendency: T=0, I=0, MM=0, final_label=0

Example 7 — NOT sarcastic: funny mismatch alone is not enough
- Text: excitedly talk about a weirdly named cute animal/pet
- Image: cute animal
- Why: The post is just playful and absurd, without a clear derisive stance or ridicule target.
- Output tendency: T=0, I=0, MM=0, final_label=0

Example 8 — NOT sarcastic: awkward situation does not always imply sarcasm
- Text: "Vl đ ổn rồi, giờ tắt live còn kịp không"
- Image: awkward/funny livestream scene
- Why: This may read as an immediate embarrassed reaction, not necessarily sarcastic mockery.
- Output tendency: T=0, I=0, MM=0, final_label=0

Return exactly this JSON schema:
{
  "labels": {
    "T": 0 or 1,
    "I": 0 or 1,
    "MM": 0 or 1
  },
  "final_label": 0 or 1 or "INVALID",
  "reasoning": {
    "text_only": "Short evidence for T", // Must be written in Vietnamese
    "image_only": "Short evidence for I", // Must be written in Vietnamese
    "multimodal": "Short evidence for MM", // Must be written in Vietnamese
    "verdict": "Short final explanation" // Must be written in Vietnamese
  },
  "has_emoji": 0 or 1
}

```

\end{document}
