from __future__ import annotations

from typing import List


def parse_query_ids(
    short_name: str, prefix: str, benchmark: str = "tpch"
) -> List[str] | None:
    if "v" not in short_name:
        return None
    qpart = short_name.split("v")[0]
    qnums = qpart[len(prefix) :].split("-")
    if len(qnums) == 1:
        return [qnums[0]]
    start_q = qnums[0]
    end_q = qnums[1]

    if benchmark == "tpch":
        start_q_int = int(start_q)
        end_q_int = int(end_q)
        qids = list(range(start_q_int, end_q_int + 1))
        return [str(qid) for qid in qids]

    if benchmark == "ceb":
        ceb_query_order = [
            "1a",
            "2a",
            "2b",
            "2c",
            "3a",
            "3b",
            "4a",
            "5a",
            "6a",
            "7a",
            "8a",
            "9a",
            "9b",
            "10a",
            "11a",
            "11b",
        ]

        def parse_qstr(query: str, is_start: bool) -> str:
            if len(query) == 1:
                assert query.isdigit()
                query = f"0{query}a"
            elif len(query) == 2:
                if query[0].isdigit() and query[1].isdigit():
                    query = f"{query}a" if is_start else f"{query}z"
                elif query[0].isdigit() and query[1].isalpha():
                    query = f"0{query}"
                else:
                    raise Exception(f"Could not parse start query {query}")
            elif len(query) == 3:
                assert query[0].isdigit() and query[1].isdigit() and query[2].isalpha()
            else:
                raise Exception(f"Could not parse start query {query}")
            return query

        start_q_norm = parse_qstr(start_q, is_start=True)
        end_q_norm = parse_qstr(end_q, is_start=False)
        queries: list[str] = []
        for query in ceb_query_order:
            query_str = f"0{query}" if len(query) == 2 else query
            if query_str >= start_q_norm and query_str <= end_q_norm:
                queries.append(query)
        return queries

    raise ValueError(f"Unknown benchmark: {benchmark}")
