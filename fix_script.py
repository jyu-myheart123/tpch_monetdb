import sys

with open("tpch_monetdb/oracle/tpch_validator.py", "r", encoding="utf-8") as f:
    content = f.read()

old_code = """        else:
            # 找不到匹配行，创建一个 mismatch（行不存在）
            mismatches.append(
                TpchCellMismatch(
                    row=expected_idx,
                    column=\"(row)\",
                    expected=expected_row,
                    actual=None,
                    diff_type=\"missing_row\",
                    message=f\"Expected row {expected_idx} has no match in actual result\",
                )
            )"""

new_code = """        else:
            if len(expected.rows) == 1 and len(actual.rows) == 1:
                actual_idx, actual_row = remaining_actual[0]
                row_mismatches = _compare_row_values(
                    row_idx=expected_idx,
                    expected_row=expected_row,
                    actual_row=actual_row,
                    columns=expected.columns,
                    contract=contract,
                )
                mismatches.extend(row_mismatches)
                remaining_actual = []
            else:
                mismatches.append(
                    TpchCellMismatch(
                        row=expected_idx,
                        column=\"(row)\",
                        expected=expected_row,
                        actual=None,
                        diff_type=\"missing_row\",
                        message=f\"Expected row {expected_idx} has no match in actual result\",
                    )
                )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open("tpch_monetdb/oracle/tpch_validator.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("File updated successfully")
else:
    print("Old code not found")
