# Tools

Internal inspection tools for diagnosing regressions without editing production code.

## `beatit_tool_probe_runner`

Run one explicit analysis window against a file and dump the head of the beat/downbeat timelines.

```bash
cmake --build build --target beatit_tool_probe_runner -j8
BEATIT_TEST_CPU_ONLY=1 ./build/beatit_tool_probe_runner training/lethal.wav 99.3216 30.0 sparse
```

## `beatit_tool_alignment_inspector`

Run the sparse windowed path for a file and print local beat-to-peak offset windows across the song.

```bash
cmake --build build --target beatit_tool_alignment_inspector -j8
BEATIT_TEST_CPU_ONLY=1 ./build/beatit_tool_alignment_inspector training/vamp.wav 32
```
