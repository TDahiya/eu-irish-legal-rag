"""Agentic retrieval: let the model search repeatedly instead of once.

Two things one-shot RAG cannot do:

  Multi-hop. "Can my spouse work on my Critical Skills permit, and does the AI Act
  apply to the model I build?" needs two unrelated retrievals - Irish immigration
  guidance and EU tech regulation. A single query embedding lands between the two and
  retrieves neither well.

  Recovery. If a query returns nothing useful, one-shot RAG answers from nothing. Here
  the model can rephrase and search again.

This module used to retry an answer whose grounding score fell below 0.45, on the
idea that a quality measurement should change the output rather than just be
displayed. Measuring it killed it. Over 36 evaluated answers grounding ran 0.68 to
0.96, so the retry never once fired; and splitting those answers at the median,
the LOWER-grounding half scored 98.2% faithfulness against 91.7% for the upper
half. The signal is inversely related to the thing it was gating on, because an
answer that paraphrases a statute in plain English scores lower on similarity than
one that copies its wording, while being just as accurate. No threshold fixes
that, so the retry is gone rather than retuned. See eval/report.md.

Citation numbering is global across every search in a turn: each tool result is
presented with its index in one growing hit list, so [n] in the final answer still
maps to hits[n-1] no matter how many searches happened.
"""
from __future__ import annotations

from rag.generate import ABSTAIN, MAX_TOKENS, MODEL, REASONING_EFFORT, SYSTEM
from rag.retrieve import LAW_TYPES, Retriever

MAX_SEARCHES = 5

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

    if not hits:
        text = ABSTAIN + " No sources matched the question."
    return text, hits
