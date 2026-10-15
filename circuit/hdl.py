"""Verilog of a DWC controller and its repair patch, for simulation and synthesis (``synthesize.py``).

The controller is the DWC hardware of Kresse & Lampert (2026b), unchanged:
the modules in ``dwc_hdl/`` (copied from the DWC repository, see its README)
with the two packages that hold a controller's network, written here from the
``DWCPolicy`` in the DWC exporter's format (``core_package``,
``threshold_package``). ``top`` wraps the controller core (``dwc_core``, the
``PIPE_STAGES`` pipeline of ``dwn_top_ooc.sv``) in ReachECO's top: registered
inputs (signed observations through ``thermo_encode_thresholded``, or
thermometer bits for the simulation check), one more register on the
popcounts (``pop_r``) and the output register.

The patch (Section 2.3, ``patch.py``) is ReachECO's own block
(``keep_hierarchy``): interval guards over the thermometer bits, one AND per
cluster, the address encoder, the correction memory (a block of its own,
``patch_memory``) and a signed adder per action, written behaviourally and
pipelined to the controller's latency, so its adder has a stage of its own.

Ports of ``top``: ``clk``, ``rstn``, the inputs and ``out``, action ``a`` at
``out[a*W +: W]``: the popcount (``W`` = its width) without a patch, the
corrected value in two's complement (``W`` = the adder width) with one.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import cost
from .guards import _guard_bits
from .patch import grid_code

ONE, ZERO = "1'b1", "1'b0"


def _popcount_width(policy):
    return cost.unsigned_width(policy.output_count)


DWC_HDL = Path(__file__).resolve().parent / "dwc_hdl"
DWC_CORE = ("mapping_layer.sv", "dwn_lut_layer.sv", "dwn_group_sum_pipelined.sv", "dwc_core.sv")
DWC_ENCODER = "thermo_encode_thresholded.sv"


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


def _layout(policy):
    """Per dimension: thermometer bits and their offset in ReachECO's code; the DWC's uniform ``BITS_PER_OBS``."""
    code = grid_code(policy)
    return [int(b) for b in code.bits], code.offsets.tolist(), max(int(b) for b in code.bits)


def core_package(policy, word) -> str:
    """``core_pkg.svh`` of the DWC exporter (``export_to_verilog``) for the policy's network.

    Dimensions with fewer thresholds than ``BITS_PER_OBS`` are padded with
    never-true thresholds (``threshold_package``); the first layer's mapping
    addresses the padded thermometer bus.
    """
    bits, offsets, width = _layout(policy)
    padded = [d * width + j for d, b in enumerate(bits) for j in range(b)]
    (first, last) = policy.mappings
    fan_in = first.shape[1]
    lines = ["`ifndef CORE_PKG_SVH", "`define CORE_PKG_SVH\n", "`timescale 1ns/1ps", "package core_pkg;\n",
             f"  parameter int OBS = {len(bits)};", f"  parameter int LUTS       = {len(first)};",
             f"  parameter int INPUTS_LUT = {fan_in};\n", f"  parameter int BITS_PER_OBS = {width};",
             f"  parameter int RAW_OBS_BIT_WIDTH = {word or 12};", "  parameter int N_BITS = INPUTS_LUT * LUTS;",
             f"  parameter int ACTIONS = {policy.action_dim};", f"  parameter int LAST_LUTS = {len(last)};",
             "  parameter int ACTION_BITWIDTH = $clog2(LAST_LUTS/ACTIONS+1); // bitwidth of action outputs",
             "  parameter int ENCODE_OBS_BIT_WIDTH = $clog2(BITS_PER_OBS+1); // bitwidth for encoding observation inputs"]
    for layer, (mapping, table) in enumerate(zip(policy.mappings, policy.luts)):
        size = "LAST_LUTS" if layer == len(policy.mappings) - 1 else "LUTS"
        rows = mapping.cpu().tolist()
        if layer == 0:
            rows = [[padded[i] for i in row] for row in rows]
        lines += ["  // Mapping array", f"  parameter int unsigned MAPPING_{layer} [0:{size}* INPUTS_LUT-1] = '{{"]
        lines += [f"    {', '.join(map(str, row))}{',' if gate < len(rows) - 1 else ''}   // LUT {gate}"
                  for gate, row in enumerate(rows)]
        lines += ["  };", f"  parameter logic [0:{size}-1][0:(1<<INPUTS_LUT)-1] LUT_INIT_{layer} = '{{"]
        truth = table.cpu().int().tolist()
        bit = "1'b{}".format
        lines += ["    '{" + ", ".join(bit(b) for b in reversed(row)) + "}" + ("," if gate < len(truth) - 1 else "")
                  + f"   // LUT {gate}" for gate, row in enumerate(truth)]
        lines += ["  };", "\n"]
    lines.append("\nendpackage\n`endif\n")
    return "\n".join(lines)


def threshold_package(policy, word) -> str:
    """``threshold_pkg.svh`` of the DWC exporter (``export_thresholds_to_svh``): ``word``-bit
    thresholds (``quantized_thresholds``), padded rows never true (the maximum)."""
    bits, _, width = _layout(policy)
    top_value = (1 << (word - 1)) - 1
    rows = [row + [top_value] * (width - len(row)) for row in quantized_thresholds(policy, word)]
    literal = lambda v: f"-{word}'sd{-v}" if v < 0 else f"{word}'sd{v}"
    lines = ["package threshold_pkg;", "", f"  localparam int OBS = {len(rows)};",
             f"  localparam int NUM_THRESHOLDS = {width};", f"  localparam int THRESH_WIDTH = {word};", "",
             "  localparam logic signed [THRESH_WIDTH-1:0] NORMALIZATION_THRESHOLDS [0:OBS-1][0:NUM_THRESHOLDS-1] = '{"]
    lines += [f"    '{{{', '.join(map(literal, row))}}}{',' if d < len(rows) - 1 else ''}" for d, row in enumerate(rows)]
    lines += ["  };", "", "endpackage : threshold_pkg", ""]
    return "\n".join(lines)


def sources(word) -> list:
    """The DWC modules a design needs, in compile order after the packages."""
    return [DWC_HDL / name for name in ((DWC_ENCODER,) if word else ()) + DWC_CORE]


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
    # ENCODING
    for dim in range(code.dims):
        for lower, upper in torch.unique(torch.stack((patch.lower[:, dim], patch.upper[:, dim]), 1), dim=0).tolist():
            low, high = _guard_bits(code, dim, lower, upper)
            terms = ([f"bits[{offsets[dim] + low}]"] if low is not None else []) + \
                    ([f"~bits[{offsets[dim] + high}]"] if high is not None else [])
            name = f"d{dim}_{lower}_{upper}"
            guards[(dim, lower, upper)] = name
            lines.append(f"  wire {name} = {' & '.join(terms) or ONE};")
    lines.append(f"  wire [{clusters - 1}:0] hit;")
    # compute the AND reduction
    for index, (lower, upper) in enumerate(zip(patch.lower.tolist(), patch.upper.tolist())):
        terms = [guards[(dim, lo, hi)] for dim, (lo, hi) in enumerate(zip(lower, upper))]
        lines.append(f"  assign hit[{index}] = {' & '.join(terms)};")
    hits = "hit"
    # register
    if delay >= 1:
        lines += _register("hit_r", clusters, "hit")
        hits = "hit_r"
    address = max(1, clusters.bit_length())
    lines.append(f"  wire [{address - 1}:0] address;")
    # COMPUTE OR Adress i.e. where the bit is high of the address
    for bit in range(address):
        terms = [f"{hits}[{c}]" for c in range(clusters) if ((c + 1) >> bit) & 1]
        lines.append(f"  assign address[{bit}] = {' | '.join(terms) or ZERO};")
    selected = "address"
    if delay >= 2:
        lines += _register("address_r", address, "address")
        selected = "address_r"
    # The correction memory is a block of its own, so its utilization is reported separately.
    # width that holds each delta change for each action
    stored = [cost.minimum_signed_width(row) for row in words.tolist()]
    offsets = [sum(stored[:action]) for action in range(len(stored))]
    lines += [f"  wire [{sum(stored) - 1}:0] words;",
              f'  (* keep_hierarchy = "yes" *) patch_memory u_memory(.address({selected}), .words(words));']
    for action, row in enumerate(words.tolist()):
        lines.append(f"  wire signed [{stored[action] - 1}:0] word{action} = words[{offsets[action]} +: {stored[action]}];")
        # delay chain
        word = f"word{action}"
        for step in range(max(0, delay - 2)):
            lines += [f"  reg signed [{stored[action] - 1}:0] {word}_{step};",
                      f"  always @(posedge clk or negedge rstn) if (!rstn) {word}_{step} <= 0; "
                      f"else {word}_{step} <= {word};"]
            word = f"{word}_{step}"
        # add
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


def top(policy, patch=None, *, word=None, stages=None) -> tuple:
    """The design's files (``core_pkg.svh``, ``threshold_pkg.svh`` with ``word``, ``design.sv``) and the
    output width per action; ``sources(word)`` are compiled between the packages and ``design.sv``.

    Without ``word`` the inputs are the thermometer bits; with it, signed
    ``word``-bit observations through the DWC's comparators. ``stages`` is the
    DWC's ``PIPE_STAGES`` (default: ``default_stages``).
    """
    stages = default_stages(policy) if stages is None else stages
    bits, offsets, width = _layout(policy)
    code_bits, pop_width = sum(bits), _popcount_width(policy)
    out = max(cost.adder_widths(policy.output_count, cost.build_words(patch.deltas))) if patch else pop_width
    total = policy.action_dim * out
    bus = len(bits) * width
    if word:
        inputs = len(bits) * word
        front = [f"module top(input logic clk, input logic rstn, input logic [{inputs - 1}:0] x_in, "
                 f"output logic [{total - 1}:0] out);",
                 *_register("x", inputs, "x_in"),
                 f"  logic signed [0:{word - 1}] quant [0:{len(bits) - 1}];",
                 *(f"  assign quant[{d}] = x[{d * word} +: {word}];" for d in range(len(bits))),
                 f"  logic [0:{bus - 1}] thermo_bus;",
                 f"  thermo_encode_thresholded #(.OBS({len(bits)}), .BITS_PER_OBS({width}), .X({word})) "
                 "u_enc (.quant_in(quant), .thermo_out(thermo_bus));",
                 f"  wire [{code_bits - 1}:0] bits;"]
        front += [f"  assign bits[{offsets[d] + j}] = thermo_bus[{d * width + j}];"
                  for d, b in enumerate(bits) for j in range(b)]
    else:
        front = [f"module top(input logic clk, input logic rstn, input logic [{code_bits - 1}:0] bits_in, "
                 f"output logic [{total - 1}:0] out);",
                 *_register("bits", code_bits, "bits_in"), f"  logic [0:{bus - 1}] thermo_bus;"]
        front += [f"  assign thermo_bus[{d * width + j}] = " + (f"bits[{offsets[d] + j}];" if j < b else "1'b0;")
                  for d, b in enumerate(bits) for j in range(width)]
    lines = [*front,
             f"  logic [0:{pop_width - 1}] group_sum [0:{policy.action_dim - 1}];",
             f"  dwc_core #(.PIPE_STAGES({stages})) u_dwc (.clk(clk), .rstn(rstn), .thermo_bus(thermo_bus), "
             ".group_sum_int(group_sum));",
             f"  wire [{policy.action_dim * pop_width - 1}:0] pop;",
             *(f"  assign pop[{a * pop_width} +: {pop_width}] = group_sum[{a}];" for a in range(policy.action_dim)),
             *_register("pop_r", policy.action_dim * pop_width, "pop"),    # output stage: the adder gets its own
             f"  wire [{total - 1}:0] value;",
             '  (* keep_hierarchy = "yes" *) patch u_patch(.clk(clk), .rstn(rstn), .bits(bits), .pop(pop_r), .value(value));'
             if patch else "  assign value = pop_r;",
             "  always @(posedge clk or negedge rstn) if (!rstn) out <= 0; else out <= value;",
             "endmodule"]
    design = "`timescale 1ns/1ps\n`include \"core_pkg.svh\"\n\n" + \
        "\n\n".join(([patch_module(policy, patch, latency(stages) + 1)] if patch else []) + ["\n".join(lines)]) + "\n"
    files = {"core_pkg.svh": core_package(policy, word), "design.sv": design}
    if word:
        files["threshold_pkg.svh"] = threshold_package(policy, word)
    return files, out


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
