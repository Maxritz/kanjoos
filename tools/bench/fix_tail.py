#!/usr/bin/env python3
import re, sys
p = "tools/bench/q4k_stream_ffn.hip"
text = open(p, encoding="utf-8").read()

pat = re.compile(r"^(\s*\{[^}]*experiment 4b[^\n]*\n(?:[^\n]*\n)*?\}[ \t]*\n(?:[ \t]*\n)*\s*HIPCHECK\(hipStreamDestroy\(compute\)\);\n(?:[ \t]*\n)*\s*printf\([^)]*Summary[^\n]*\n(?:[ \t]*\n)*\s*printf\([^)]*checks failed[^\n]*\n(?:[ \t]*\n)*\s*printf\([^)]*RESULT[^\n]*\n(?:[ \t]*\n)*\s*return g_failed == 0 \? 0 : 5;\n\}\n)", re.M)
m = pat.search(text)
if not m:
    print("ERROR: tail block not found", file=sys.stderr)
    sys.exit(1)
block = m.group(1)
clean = (
"    {   // experiment 4b: the WAITING consumer, where the deadline is real.\n"
"        // Same stack, same slot budget, demand (no fallback), steps block on\n"
"        // wait_ready; the manager's clock advances with every spin. The sweep\n"
"        // {0, 25, 200, 5000} measures the C9 miss curve and its wall impact.\n"
"        const int deadlines[4] = {0, 25, 200, 5000};\n"
"        for (int di = 0; di < 4; ++di) {\n"
"            StackStats ds;\n"
"            PassCfg pc{\"wait sweep\", &slotsMain, 0, 0, false, deadlines[di], false, DT, DNL};\n"
"            pc.waitmode = true;\n"
"            run_stack(pc, dpl, dxin, dG2, dU2, dY2, ds);\n"
"            printf(\"\\n--- pass 4b: wait sweep deadline=%d ticks ---\\n\", deadlines[di]);\n"
"            printf(\"  wall %.3f ms  hand-offs %llu  C21 misses %llu\\n\",\n"
"                   (float)ds.wall_ms, (unsigned long long)ds.handoffs,\n"
"                   (unsigned long long)ds.agg.misses);\n"
"            for (int L = 0; L < DNL; ++L) {\n"
"                std::vector<double> ref((size_t)DT * TOPK * KH);\n"
"                for (size_t i = 0; i < ref.size(); ++i) ref[i] = (double)dY2[(size_t)L][i];\n"
"                char nm[64];\n"
"                std::snprintf(nm, sizeof(nm), \"waitsweep down L%d\", L);\n"
"                cx(nm, ds.Yout[(size_t)L], ref, DT * TOPK, KH, 1e-4);\n"
"            }\n"
"            report(pc, ds);\n"
"        }\n"
"    }\n"
"\n"
"    HIPCHECK(hipStreamDestroy(compute));\n"
"\n"
"    printf(\"\\n=== Summary ===\\n\");\n"
"    printf(\"  checks failed: %d\\n\", g_failed);\n"
"    printf(\"\\nRESULT: %s\\n\", g_failed == 0 ? \"PASS\" : \"FAIL\");\n"
"    return g_failed == 0 ? 0 : 5;\n"
"}\n"
)
open(p, "w", encoding="utf-8").write(text.replace(block, clean, 1))
print("rewrote tail block OK")
