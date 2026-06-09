def truncate_model_final_output(
    log: str, num_keep_lines_per_code_block: int = 20
) -> str:
    begin_tag = "<BEGIN_FILES>"
    end_tags = ["<END_FILES>", "</END_FILES>"]
    result_parts: list[str] = []
    pos = 0
    while True:
        begin_idx = log.find(begin_tag, pos)
        if begin_idx == -1:
            result_parts.append(log[pos:])
            break
        end_idxs = []
        for end_tag in end_tags:
            idx = log.find(end_tag, begin_idx)
            if idx != -1:
                end_idxs.append((idx, end_tag))
        end_idx = min(end_idxs, key=lambda item: item[0])[0] if end_idxs else -1
        end_tag = min(end_idxs, key=lambda item: item[0])[1] if end_idxs else None
        if end_idx == -1:
            result_parts.append(log[pos:])
            break
        result_parts.append(log[pos:begin_idx])
        assert end_tag is not None, "End tag should not be None if end_idx is valid"
        between = log[begin_idx : end_idx + len(end_tag)]
        between_lines = between.splitlines()
        if len(between_lines) > num_keep_lines_per_code_block:
            truncated_between = (
                "\n".join(between_lines[:num_keep_lines_per_code_block])
                + "\n...[truncated]...\n"
            )
        else:
            truncated_between = between
        result_parts.append(truncated_between)
        pos = end_idx + len(end_tag)
    return "".join(result_parts)

