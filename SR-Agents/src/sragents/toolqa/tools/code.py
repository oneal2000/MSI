"""Code execution tools for ToolQA: PythonInterpreter, SQLInterpreter.

PythonInterpreter: subprocess with 'ans' variable capture.
SQLInterpreter: sqlite3 in-memory database.
"""

import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time


# Match the Python tool execution budget; this also applies in worker threads.
SQL_QUERY_TIMEOUT_SECONDS = 30.0


def python_interpret(code: str) -> str:
    """Execute Python code in a subprocess and return the value of 'ans'.

    Runs in a temporary directory via subprocess to:
    1. Prevent file pollution in the project root (e.g. sqlite3 databases)
    2. Be thread-safe (no os.chdir which is process-global)
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        script = os.path.join(tmp_dir, "script.py")
        with open(script, "w") as f:
            f.write("ans = 0\n")
            f.write(code)
            f.write("\nprint(ans)\n")
        result = subprocess.run(
            [sys.executable, script],
            cwd=tmp_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip())
        return result.stdout.strip()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def sql_interpret(sql_cmd: str, conn: sqlite3.Connection) -> str:
    """Execute SQL with a cancellable budget, preserving the tool output format.

    The SQLite progress handler interrupts execution itself, including result
    fetching; a timeout on a Python future would leave the query running.
    """
    import re
    translated = re.sub(r"(\w+)\.(\w+_data)\b", r"\1_data", sql_cmd)
    deadline = time.monotonic() + SQL_QUERY_TIMEOUT_SECONDS
    timed_out = False

    def expired() -> bool:
        nonlocal timed_out
        timed_out = time.monotonic() >= deadline
        return timed_out

    cursor = conn.cursor()
    conn.set_progress_handler(expired, 10_000)
    try:
        cursor.execute(translated)
        if cursor.description is None:
            return "Query executed successfully."
        column_names = [desc[0] for desc in cursor.description]
        rows_string = []
        for row in cursor:
            if expired():
                raise TimeoutError(f"SQL query exceeded {SQL_QUERY_TIMEOUT_SECONDS:g} seconds")
            rows_string.append(", ".join(
                f"{column_names[i]}: {row[i]}" for i in range(len(row))
            ))
        return "\n".join(rows_string)
    except sqlite3.OperationalError as exc:
        if timed_out:
            raise TimeoutError(
                f"SQL query exceeded {SQL_QUERY_TIMEOUT_SECONDS:g} seconds"
            ) from exc
        raise
    finally:
        conn.set_progress_handler(None, 0)
        cursor.close()
