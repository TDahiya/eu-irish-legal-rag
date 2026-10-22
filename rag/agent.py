"""Agentic retrieval: let the model search repeatedly, then check its own grounding.

Two things one-shot RAG cannot do:

  Multi-hop. "Can my spouse work on my Critical Skills permit, and does the AI Act
  apply to the model I build?" needs two unrelated retrievals - Irish immigration
  guidance and EU tech regulation. A single query embedding lands between the two and
  retrieves neither well.

  Recovery. If a query returns nothing useful, one-shot RAG answers from nothing. Here
  the model can rephrase and search again.

Then self-correction: the answer's grounding score is computed and, if it is poor, the
answer is retried once with the accumulated context. The evaluation layer stops being a
dashboard and becomes a control signal.

Citation numbering is global across every search in a turn: each tool result is
presented with its index in one growing hit list, so [n] in the final answer still
maps to hits[n-1] no matter how many searches happened.
"""
from __future__ import annotations

from rag.generate import (ABSTAIN, MAX_TOKENS, MODEL, REASONING_EFFORT,
                          SYSTEM, answer)
from rag.retrieve import LAW_TYPES, Retriever
from rag.score import live_scores

MAX_SEARCHES = 5
GROUNDING_FLOOR = 0.45      # below this, the answer is retried once

def build_tools(source_types: tuple[str, ...]) -> list[dict]:
    """The tool schema, with the enum restricted to the types this turn may search.

    Built per call rather than as a constant so a source filter chosen in the UI
    actually constrains the agent, instead of being a control that does nothing.
    """
    return [{
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the corpus of EU and Irish law. Call it once per distinct "
                "sub-question: a question spanning immigration and technology law "
                "needs a separate search for each. Rephrase and search again if "
                "results look unhelpful. Use exact legal vocabulary and article or "
                "section numbers when the question contains them."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "The search query."},
                    "source_type": {
                        "type": "string",
                        "enum": [*source_types, "any"],
                        "description": (
                            "Narrow the search: eu_law for EU regulations and "
                            "directives, ie_legislation for Acts of the Oireachtas, "
                            "ie_guidance for Irish immigration policy and procedure, "
                            "case_law for CJEU judgments, philosophy for "
                            "public-domain books on law and justice. 'any' searches "
                            "everything available."),
                    },
                },
                "required": ["query"],
            },
        },
    }]

AGENT_SYSTEM = SYSTEM + """

You have a `search` tool. Search before you answer, once per distinct sub-question.
Never answer from your own knowledge of the law; if search returns nothing useful
after you have rephrased, abstain. The source numbers in tool results are stable for
the whole conversation - cite them as [n]."""


def _render(hits: list[dict], offset: int) -> str:
    """Tool output, numbered with each hit's global index."""
    if not hits:
        return "No results. Try different wording or a different source_type."
    out = []
    for i, h in enumerate(hits, offset + 1):
        head = "[%d] %s" % (i, h["citation"])
        if h.get("note"):
            head += "  (NOTE: %s)" % h["note"]
        elif h.get("ireland_bound") == 0:
            head += "  (NOTE: Ireland is not bound by this instrument.)"
        out.append("%s\n%s" % (head, h["text"]))
    return "\n\n".join(out)


def run(question: str, retriever: Retriever, client, model: str = MODEL,
        k: int = 5, trace: list | None = None,
        source_types: tuple[str, ...] = LAW_TYPES) -> tuple[str, list[dict]]:
    """Returns (answer_text, hits) with hits in citation order."""
    messages = [{"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": question}]
    tools = build_tools(source_types)
    hits: list[dict] = []
    seen: set[int] = set()
    searches = 0

    while True:
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=tools, temperature=0,
            max_completion_tokens=MAX_TOKENS, reasoning_effort=REASONING_EFFORT,
        )
        msg = resp.choices[0].message
        calls = msg.tool_calls or []

        if not calls:
            text = msg.content or ""
            break

        messages.append(msg)
        for call in calls:
            import json
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            query = args.get("query") or question
            st = args.get("source_type")
            # A model-chosen type outside what this turn is allowed falls back to
            # the whole allowed set rather than searching something it may not.
            types = (st,) if st in source_types else source_types

            if searches >= MAX_SEARCHES:
                result = "Search limit reached. Answer from the sources you have, or abstain."
            else:
                searches += 1
                fresh = [h for h in retriever.search(query, k=k, source_types=types)
                         if h["id"] not in seen]
                offset = len(hits)
                for h in fresh:
                    seen.add(h["id"])
                hits.extend(fresh)
                result = _render(fresh, offset)
                if trace is not None:
                    trace.append({"query": query, "source_type": st or "any",
                                  "n_new": len(fresh)})

            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": result})

    # ---- self-correction: the eval signal decides whether to try again
    if hits and not text.strip().startswith(ABSTAIN[:40]):
        scores = live_scores(text, hits, retriever.model)
        if scores["grounding"] is not None and scores["grounding"] < GROUNDING_FLOOR:
            if trace is not None:
                trace.append({"retry": "grounding %.2f < %.2f"
                                       % (scores["grounding"], GROUNDING_FLOOR)})
            retry = answer(question, hits, model=model, client=client)
            retry_scores = live_scores(retry, hits, retriever.model)
            # keep the retry only if it is actually better grounded
            if (retry_scores["grounding"] or 0) > scores["grounding"]:
                text = retry

    if not hits:
        text = ABSTAIN + " No sources matched the question."
    return text, hits
