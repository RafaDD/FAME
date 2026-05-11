from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING, Any, List

from openai import OpenAI
from google import genai
from google.genai import types

from arxiv_agent.topic_embedding import embed_texts
from schemas import GeneratedIdea
from arxiv_agent.run import _progress

if TYPE_CHECKING:
    from config import AppConfig


# ── LLM-as-judge evaluation ──────────────────────────────────────────────────


def _extract_json_payload(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _fallback_evaluations(ideas: List[GeneratedIdea]) -> List[dict]:
    evaluations: list[dict] = []
    for idea in ideas:
        novelty = 7 if idea.abstract else 6
        feasibility = 7 if idea.abstract else 5
        interestingness = 7 if idea.abstract else 6
        potential_impact = 7 if idea.abstract else 6
        evaluations.append(
            {
                "novelty_score": novelty,
                "feasibility_score": feasibility,
                "interestingness_score": interestingness,
                "potential_impact_score": potential_impact,
                "evaluation_notes": (
                    "Fallback heuristic evaluation used due to unavailable evaluator output."
                ),
            }
        )
    return evaluations


def _coerce_score(value: Any, default: int = 5) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(1, min(10, score))


def evaluate_ideas(
    config: "AppConfig",
    ideas: List[GeneratedIdea],
    hint_papers: list[tuple[str, float]] | None = None,
) -> List[GeneratedIdea]:
    eval_model = config.llm.eval_model or os.getenv("OPENAI_EVAL_MODEL", "gpt-4.1")
    _is_google = "gemini" in eval_model.lower()

    api_key = os.getenv("OPENAI_API_KEY")
    if not ideas:
        return ideas

    if not api_key:
        evaluations = _fallback_evaluations(ideas)
        for idea, ev in zip(ideas, evaluations):
            idea.novelty_score = ev["novelty_score"]
            idea.feasibility_score = ev["feasibility_score"]
            idea.interestingness_score = ev["interestingness_score"]
            idea.potential_impact_score = ev["potential_impact_score"]
            idea.evaluation_notes = ev["evaluation_notes"]
        return ideas

    _progress(f"using {eval_model} for idea evaluation")

    ideas_payload = [idea.model_dump(mode="json") for idea in ideas]
    topic = config.pipeline.topic

    hint_section = ""
    if hint_papers:
        # Sort by weight descending and take top papers
        sorted_hints = sorted(hint_papers, key=lambda x: x[1], reverse=True)[:15]
        hint_lines = [
            f"- \"{text}\" (impact weight: {weight:.2f})" for text, weight in sorted_hints
        ]
        hint_section = (
            "\nReference: Use the following papers published after your knowledge cutoff as reference to calibrate your scoring of novelty and potential impact:\n"
            + "\n".join(hint_lines)
            + "\n"
        )

    prompt = f"""
You are a strict research idea evaluator.
Topic: {topic}
{hint_section}
For each idea, provide:

- novelty_score (1–10): Originality relative to existing literature; is it non-obvious and new?
- interestingness_score (1–10): Intellectual appeal and whether it raises compelling questions or insights
- potential_impact_score (1–10): Expected influence on the field if successful (e.g., advancing theory, shifting paradigms, enabling new research directions, or widely adopted applications)

Scoring Guidelines:

- Use only integers from 1 (very low) to 10 (exceptional)
- Be conservative; avoid inflated scores
- Compare against typical research in the field, not absolute ideals

Return concise evaluation_notes (1-3 sentences) per idea.
Output strict JSON only:
{{
  "evaluations": [
    {{
      "index": 0,
      "novelty_score": 7,
      "interestingness_score": 8,
      "potential_impact_score": 7,
      "evaluation_notes": "..."
    }}
  ]
}}

Ideas to evaluate:
{json.dumps(ideas_payload, indent=2)}
"""
    try:
        if _is_google:
            google_client = genai.Client(
                api_key=api_key,
                http_options={
                    "base_url": os.getenv("GOOGLE_BASE_URL"),
                },
            )
            response = google_client.models.generate_content(
                model=eval_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1),
            )
            output_text = response.text.strip() if response.text else ""
        else:
            openai_client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL"),
            )
            response = openai_client.chat.completions.create(
                model=eval_model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            output_text = response.choices[0].message.content.strip()

        payload = _extract_json_payload(output_text)
        evaluations = payload.get("evaluations", [])
        if not isinstance(evaluations, list):
            raise ValueError("Invalid evaluations payload.")
    except Exception as exc:
        print(f"[utils] Idea evaluation failed: {exc}. Using fallback evaluation.", flush=True)
        evaluations = _fallback_evaluations(ideas)
        for idea, ev in zip(ideas, evaluations):
            idea.novelty_score = ev["novelty_score"]
            idea.feasibility_score = ev["feasibility_score"]
            idea.interestingness_score = ev["interestingness_score"]
            idea.potential_impact_score = ev["potential_impact_score"]
            idea.evaluation_notes = ev["evaluation_notes"]
        return ideas

    eval_by_idx: dict[int, dict] = {}
    for item in evaluations:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if isinstance(idx, int):
            eval_by_idx[idx] = item

    for idx, idea in enumerate(ideas):
        item = eval_by_idx.get(idx)
        if not item:
            fallback = _fallback_evaluations([idea])[0]
            idea.novelty_score = fallback["novelty_score"]
            idea.feasibility_score = fallback["feasibility_score"]
            idea.interestingness_score = fallback["interestingness_score"]
            idea.potential_impact_score = fallback["potential_impact_score"]
            idea.evaluation_notes = fallback["evaluation_notes"]
            continue
        idea.novelty_score = _coerce_score(item.get("novelty_score"))
        idea.feasibility_score = _coerce_score(item.get("feasibility_score"))
        idea.interestingness_score = _coerce_score(item.get("interestingness_score"))
        idea.potential_impact_score = _coerce_score(item.get("potential_impact_score"))
        idea.evaluation_notes = str(item.get("evaluation_notes", "")).strip() or None
    return ideas


# ── Manifold-based idea scoring ───────────────────────────────────────────────


def assign_manifold_idea_scores(
    ideas: List[GeneratedIdea],
    manifold_result: Any,
    config: "AppConfig",
) -> None:
    if not ideas or manifold_result is None:
        return
    try:
        import torch
        from manifold.analysis import frontier_scores

        embedding_model = config.llm.embedding_model

        idea_texts = [f"{idea.title}\n\n{idea.abstract}".strip() for idea in ideas]
        idea_embeddings, _ = embed_texts(idea_texts, model=embedding_model)

        trainer = manifold_result.trainer
        dataset = manifold_result.dataset

        x_ideas = torch.tensor(idea_embeddings, dtype=torch.float32)
        t_ideas = torch.ones(len(ideas), dtype=torch.float32)

        device = torch.device("cpu")
        mapper = trainer.mapper.to(device)
        spine = trainer.spine.to(device)
        mapper.eval()
        spine.eval()

        with torch.no_grad():
            z_ideas = mapper(x_ideas, t_ideas)

        K = dataset.n_clusters
        k_idx = torch.arange(K, dtype=torch.long)
        t_now_vec = torch.ones(K, dtype=torch.float32)
        with torch.no_grad():
            mu_now = spine(k_idx, t_now_vec)
        dists = torch.cdist(z_ideas, mu_now)
        idea_cluster_ids = dists.argmin(dim=1)

        idea_ids = [f"__idea_{i}__" for i in range(len(ideas))]

        fr_ideas, _ = frontier_scores(
            z=z_ideas,
            spine=spine,
            times=t_ideas,
            cluster_ids=idea_cluster_ids,
            arxiv_ids=idea_ids,
            t_now=1.0,
            recent_fraction=1.0,
        )

        z_papers = manifold_result.z
        t_papers = trainer.t.cpu()
        c_papers = trainer.c.cpu()
        paper_ids = dataset.arxiv_ids

        fr_papers, _ = frontier_scores(
            z=z_papers, spine=spine, times=t_papers,
            cluster_ids=c_papers, arxiv_ids=paper_ids, t_now=1.0,
        )

        fr_ref = [v for v in fr_papers.values() if not math.isinf(v)]
        fr_all_finite = fr_ref + list(fr_ideas.values())
        if fr_all_finite:
            fr_min = min(fr_all_finite); fr_max = max(fr_all_finite)
            fr_range = fr_max - fr_min if abs(fr_max - fr_min) > 1e-12 else 1.0
        else:
            fr_min, fr_range = 0.0, 1.0

        for i, idea in enumerate(ideas):
            aid = idea_ids[i]
            idea.manifold_frontier_score = fr_ideas[aid]

        print("[utils] Manifold idea scoring complete.", flush=True)
    except Exception as exc:
        print(f"[utils] Manifold idea scoring failed ({exc}); skipping.", flush=True)
