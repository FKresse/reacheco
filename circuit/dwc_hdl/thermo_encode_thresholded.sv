`timescale 1ns/1ps
`include "core_pkg.svh"
`include "threshold_pkg.svh"

module thermo_encode_thresholded #(
  parameter int OBS           = core_pkg::OBS,
  parameter int BITS_PER_OBS  = core_pkg::BITS_PER_OBS,
  parameter int X             = core_pkg::ENCODE_OBS_BIT_WIDTH,
  parameter int THRESH_WIDTH  = threshold_pkg::THRESH_WIDTH
)(
  input  logic signed [0:X-1] quant_in [0:OBS-1],
  output logic        [0:OBS*BITS_PER_OBS-1] thermo_out
);

  import core_pkg::*;
  import threshold_pkg::*;

  genvar i, j;
  generate
    for (i = 0; i < OBS; i++) begin : G_OBS
      for (j = 0; j < BITS_PER_OBS; j++) begin : G_BIT
        assign thermo_out[i*BITS_PER_OBS + j] =
          ($signed(quant_in[i]) > $signed(NORMALIZATION_THRESHOLDS[i][j]));
      end
    end
  endgenerate

endmodule