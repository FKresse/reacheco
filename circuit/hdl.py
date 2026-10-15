"""Verilog of a DWC controller and its repair patch, for simulation and synthesis (``synthesize.py``).

The controller follows the DWC hardware of Kresse & Lampert (2026b),
``dwn_top_ooc.sv``: registered inputs (thermometer bits, or signed observations
through threshold comparators, ``encoder``), one LUT primitive per gate (its
truth table as ``INIT``), a popcount per action (``dwn_group_sum``) and a
registered output, with the same ``PIPE_STAGES`` pipeline (``latency``) and one
more register on the popcounts (``pop_r``), so the patch's adder has a stage of
its own. The
patch (Section 2.3, ``patch.py``) reads the same thermometer bits: interval
guards, one AND per cluster, the address encoder, the correction memory and a
signed adder per action, written behaviourally so synthesis can pack them, and
pipelined to the controller's latency; it is a separate block
(``keep_hierarchy``), so its utilization is its cost.

Ports of ``top``: ``clk``, ``rstn``, the inputs and ``out``, action ``a`` at
``out[a*W +: W]``: the popcount (``W`` = its width) without a patch, the
corrected value in two's complement (``W`` = the adder width) with one.
"""
from __future__ import annotations

import torch

from . import cost
from .guards import _guard_bits
from .patch import grid_code

ONE, ZERO = "1'b1", "1'b0"


def _popcount_width(policy):
    return cost.unsigned_width(policy.output_count)


CHUNK = 128      # dwn_group_sum's CHUNK_SIZE


def default_stages(policy):
    """``PIPE_STAGES`` for every controller: 3 (registered LUT inputs and layer outputs, and the
    popcount in registered 128-bit chunks)."""
    return 3


def latency(stages):
    """Cycles from the thermometer bits to the popcounts ``pop`` (``dwn_top_ooc.sv``); ``top``
    registers them once more (``pop_r``) before the output."""
    sums = stages - 2 if stages > 2 else 0
    return 2 * (stages > 2) + (stages >= 1) + (stages >= 2) + (sums >= 1) + (sums >= 2)


def _register(name, width, value, reset=True):
    """Lines of a register ``name`` loaded from ``value`` (active-low asynchronous reset, as the DWC)."""
    if reset:
        return [f"  reg [{width - 1}:0] {name};",
                f"  always @(posedge clk or negedge rstn) if (!rstn) {name} <= 0; else {name} <= {value};"]
    return [f"  reg [{width - 1}:0] {name};", f"  always @(posedge clk) {name} <= {value};"]


def controller(policy, stages) -> str:
    """Module ``dwc``: ``bits`` to the popcounts ``pop`` (``P`` bits per action), ``stages`` as ``PIPE_STAGES``."""
    code = grid_code(policy)
    width, groups = _popcount_width(policy), policy.output_count
    lines = [f"module dwc(input clk, input rstn, input [{code.input_size - 1}:0] bits, "
             f"output [{policy.action_dim * width - 1}:0] pop);"]
    source = "bits"
    for layer, (mapping, table) in enumerate(zip(policy.mappings, policy.luts)):
        mapping, table = mapping.cpu(), table.cpu()
        fan_in, gates = mapping.shape[1], len(mapping)
        lines.append(f"  wire [{gates - 1}:0] l{layer};")
        for gate, (inputs, truth) in enumerate(zip(mapping.tolist(), table.tolist())):
            selected = [f"{source}[{index}]" for index in inputs]
            if stages > 2:      # luts_mapped_to_layer<l>_r
                lines += _register(f"m{layer}_{gate}", fan_in, "{" + ", ".join(reversed(selected)) + "}")
                selected = [f"m{layer}_{gate}[{pin}]" for pin in range(fan_in)]
            init = sum(1 << address for address, value in enumerate(truth) if value)
            pins = ", ".join(f".I{pin}({signal})" for pin, signal in enumerate(selected))
            lines.append(f"  LUT{fan_in} #(.INIT({1 << fan_in}'h{init:0{(1 << fan_in) // 4}x})) "
                         f"g{layer}_{gate} (.O(l{layer}[{gate}]), {pins});")
        source = f"l{layer}"
        if stages >= layer + 1:     # internal_bits_layer0_r, output_bits_r
            lines += _register(f"r{layer}", gates, source)
            source = f"r{layer}"
    sums = stages - 2 if stages > 2 else 0
    lines += [f"  function [{width - 1}:0] popcount(input [{groups - 1}:0] v);",
              "    integer i;",
              f"    begin popcount = 0; for (i = 0; i < {groups}; i = i + 1) popcount = popcount + v[i]; end",
              "  endfunction"]
    for action in range(policy.action_dim):
        bits = f"{source}[{action * groups} +: {groups}]"
        if not sums:
            lines.append(f"  assign pop[{action * width} +: {width}] = popcount({bits});")
            continue
        # dwn_group_sum: registered popcounts of 128-bit chunks, their sum, optionally registered.
        chunks = [(low, min(CHUNK, groups - low)) for low in range(0, groups, CHUNK)]
        partial = cost.unsigned_width(CHUNK)
        for index, (low, size) in enumerate(chunks):
            lines += _register(f"p{action}_{index}", partial,
                               " + ".join(f"{source}[{action * groups + low + b}]" for b in range(size)), reset=False)
        total = " + ".join(f"p{action}_{index}" for index in range(len(chunks)))
        if sums >= 2:
            lines += _register(f"s{action}", width, total, reset=False)
            total = f"s{action}"
        lines.append(f"  assign pop[{action * width} +: {width}] = {total};")
    lines.append("endmodule")
    return "\n".join(lines)


def patch_module(policy, patch, delay) -> str:
    """Module ``patch``: ``bits`` and ``pop`` to the corrected values ``value`` (``W`` bits per action).

    The correction reaches the adder ``delay`` cycles after ``bits``, together
    with the popcount: the hits, then the address, then the words are registered.
    """
    code = grid_code(policy)
    offsets = code.offsets.tolist()
    width, words = _popcount_width(policy), cost.build_words(patch.deltas)
    out = max(cost.adder_widths(policy.output_count, words))
    clusters = len(patch)
    lines = [f"module patch(input clk, input rstn, input [{code.input_size - 1}:0] bits, "
             f"input [{policy.action_dim * width - 1}:0] pop, output [{policy.action_dim * out - 1}:0] value);"]
    guards = {}
    for dim in range(code.dims):
        for lower, upper in torch.unique(torch.stack((patch.lower[:, dim], patch.upper[:, dim]), 1), dim=0).tolist():
            low, high = _guard_bits(code, dim, lower, upper)
            terms = ([f"bits[{offsets[dim] + low}]"] if low is not None else []) + \
                    ([f"~bits[{offsets[dim] + high}]"] if high is not None else [])
            name = f"d{dim}_{lower}_{upper}"
            guards[(dim, lower, upper)] = name
            lines.append(f"  wire {name} = {' & '.join(terms) or ONE};")
    lines.append(f"  wire [{clusters - 1}:0] hit;")
    for index, (lower, upper) in enumerate(zip(patch.lower.tolist(), patch.upper.tolist())):
        terms = [guards[(dim, lo, hi)] for dim, (lo, hi) in enumerate(zip(lower, upper))]
        lines.append(f"  assign hit[{index}] = {' & '.join(terms)};")
    hits = "hit"
    if delay >= 1:
        lines += _register("hit_r", clusters, "hit")
        hits = "hit_r"
    address = max(1, clusters.bit_length())
    lines.append(f"  wire [{address - 1}:0] address;")
    for bit in range(address):
        terms = [f"{hits}[{c}]" for c in range(clusters) if ((c + 1) >> bit) & 1]
        lines.append(f"  assign address[{bit}] = {' | '.join(terms) or ZERO};")
    selected = "address"
    if delay >= 2:
        lines += _register("address_r", address, "address")
        selected = "address_r"
    # The correction memory is a block of its own, so its utilization is reported separately.
    stored = [cost.minimum_signed_width(row) for row in words.tolist()]
    offsets = [sum(stored[:action]) for action in range(len(stored))]
    lines += [f"  wire [{sum(stored) - 1}:0] words;",
              f'  (* keep_hierarchy = "yes" *) patch_memory u_memory(.address({selected}), .words(words));']
    for action, row in enumerate(words.tolist()):
        lines.append(f"  wire signed [{stored[action] - 1}:0] word{action} = words[{offsets[action]} +: {stored[action]}];")
        word = f"word{action}"
        for step in range(max(0, delay - 2)):
            lines += [f"  reg signed [{stored[action] - 1}:0] {word}_{step};",
                      f"  always @(posedge clk or negedge rstn) if (!rstn) {word}_{step} <= 0; "
                      f"else {word}_{step} <= {word};"]
            word = f"{word}_{step}"
        lines += [f"  wire signed [{out - 1}:0] base{action} = {{{{{out - width}{{1'b0}}}}, pop[{action * width} +: {width}]}};",
                  f"  assign value[{action * out} +: {out}] = base{action} + {word};"]
    lines.append("endmodule")
    memory = [f"module patch_memory(input [{address - 1}:0] address, output reg [{sum(stored) - 1}:0] words);",
              "  always @* begin", "    words = 0;", "    case (address)"]
    for entry in range(1, words.shape[1]):
        fields = [f"words[{offsets[action]} +: {stored[action]}] = {int(words[action, entry])};"
                  for action in range(words.shape[0]) if int(words[action, entry])]
        if fields:
            memory.append(f"      {entry}: begin {' '.join(fields)} end")
    memory += ["    endcase", "  end", "endmodule"]
    return "\n".join(lines) + "\n\n" + "\n".join(memory)


def quantized_thresholds(policy, word):
    """Signed ``word``-bit thresholds per dimension: the grid bounds ``[lower, upper]`` map to
    ``+-(2^(word-1) - 1)``; a threshold below the grid becomes the minimum (its bit is always
    one) and one at or above the upper bound the maximum (always zero), as the policy clamps."""
    top_value = (1 << (word - 1)) - 1
    rows = []
    for row, lower, upper in zip(policy.thresholds, policy.lower.tolist(), policy.upper.tolist()):
        centre, step = (lower + upper) / 2, (upper - lower) / (2 * top_value)
        rows.append([-top_value - 1 if t < lower else top_value if t >= upper
                     else max(-top_value, min(top_value, round((t - centre) / step))) for t in row[:-1].tolist()])
    return rows


def encoder(policy, word) -> list:
    """Lines of ``top`` turning the registered inputs ``x`` (``word`` bits per dimension) into ``bits``."""
    lines, index = [], 0
    for dim, row in enumerate(quantized_thresholds(policy, word)):
        for threshold in row:
            literal = f"-{word}'sd{-threshold}" if threshold < 0 else f"{word}'sd{threshold}"
            lines.append(f"  assign bits[{index}] = $signed(x[{dim * word} +: {word}]) > {literal};")
            index += 1
    return lines


def top(policy, patch=None, *, word=None, stages=None) -> tuple:
    """The design (``top``, ``dwc`` and, with a patch, ``patch``) and the output width per action.

    Without ``word`` the inputs are the thermometer bits; with it, signed
    ``word``-bit observations through comparators (``encoder``). ``stages``
    is the DWC's ``PIPE_STAGES`` (default: ``default_stages``).
    """
    code = grid_code(policy)
    stages = default_stages(policy) if stages is None else stages
    width = _popcount_width(policy)
    out = max(cost.adder_widths(policy.output_count, cost.build_words(patch.deltas))) if patch else width
    total = policy.action_dim * out
    if word:
        inputs = len(policy.thresholds) * word
        front = [f"module top(input clk, input rstn, input [{inputs - 1}:0] x_in, output reg [{total - 1}:0] out);",
                 *_register("x", inputs, "x_in"), f"  wire [{code.input_size - 1}:0] bits;", *encoder(policy, word)]
    else:
        front = [f"module top(input clk, input rstn, input [{code.input_size - 1}:0] bits_in, "
                 f"output reg [{total - 1}:0] out);", *_register("bits", code.input_size, "bits_in")]
    lines = [*front,
             f"  wire [{policy.action_dim * width - 1}:0] pop;",
             f"  wire [{total - 1}:0] value;",
             "  dwc u_dwc(.clk(clk), .rstn(rstn), .bits(bits), .pop(pop));",
             *_register("pop_r", policy.action_dim * width, "pop"),     # output stage: the adder gets its own
             '  (* keep_hierarchy = "yes" *) patch u_patch(.clk(clk), .rstn(rstn), .bits(bits), .pop(pop_r), .value(value));'
             if patch else "  assign value = pop_r;",
             "  always @(posedge clk or negedge rstn) if (!rstn) out <= 0; else out <= value;",
             "endmodule"]
    modules = [controller(policy, stages)] + ([patch_module(policy, patch, latency(stages) + 1)] if patch else []) \
        + ["\n".join(lines)]
    return "`timescale 1ns/1ps\n\n" + "\n\n".join(modules) + "\n", out


def testbench(cells, bits, out_width, delay=0) -> str:
    """Module ``tb``: drives ``stimulus.mem`` through ``top`` (``delay`` pipeline cycles) and compares with ``expected.mem``."""
    return f"""`timescale 1ns/1ps
module tb;
  reg clk = 0;
  always #5 clk = ~clk;
  reg [{bits - 1}:0] stimulus [0:{cells - 1}];
  reg [{out_width - 1}:0] expected [0:{cells - 1}];
  reg [{bits - 1}:0] bits_in;
  wire [{out_width - 1}:0] out;
  reg rstn = 0;
  integer i, errors;
  top dut(.clk(clk), .rstn(rstn), .bits_in(bits_in), .out(out));
  initial begin
    $readmemh("stimulus.mem", stimulus);
    $readmemh("expected.mem", expected);
    errors = 0;
    #2 rstn = 1;
    for (i = 0; i < {cells}; i = i + 1) begin
      bits_in = stimulus[i];
      repeat ({2 + delay}) @(posedge clk);
      #1;
      if (out !== expected[i]) begin
        errors = errors + 1;
        if (errors <= 5) $display("MISMATCH cell %0d: %h, expected %h", i, out, expected[i]);
      end
    end
    $display("CHECKED %0d ERRORS %0d", {cells}, errors);
    $finish;
  end
endmodule
"""


def vectors(bits, values, out) -> tuple:
    """``stimulus.mem`` and ``expected.mem`` contents: thermometer codes and values (``out`` bits per action)."""
    stimulus = "\n".join(f"{sum(1 << j for j, bit in enumerate(row) if bit):x}" for row in bits.tolist())
    mask = (1 << out) - 1
    expected = "\n".join(f"{sum((int(v) & mask) << (a * out) for a, v in enumerate(row)):x}" for row in values.tolist())
    return stimulus + "\n", expected + "\n"
