`timescale 1ns/1ps
`include "core_pkg.svh"

module patch(input clk, input rstn, input [55:0] bits, input [17:0] pop, output [19:0] value);
  wire d0_3_3 = bits[2] & ~bits[3];
  wire d0_4_4 = bits[3] & ~bits[4];
  wire d0_5_5 = bits[4] & ~bits[5];
  wire d0_7_7 = bits[6] & ~bits[7];
  wire d0_8_8 = bits[7] & ~bits[8];
  wire d0_9_9 = bits[8] & ~bits[9];
  wire d1_5_5 = bits[23] & ~bits[24];
  wire d1_6_6 = bits[24] & ~bits[25];
  wire d1_9_9 = bits[27] & ~bits[28];
  wire d1_10_10 = bits[28] & ~bits[29];
  wire d1_11_11 = bits[29] & ~bits[30];
  wire d2_5_5 = bits[42] & ~bits[43];
  wire d2_6_6 = bits[43] & ~bits[44];
  wire d2_7_7 = bits[44] & ~bits[45];
  wire d3_1_1 = bits[47] & ~bits[48];
  wire d3_2_2 = bits[48] & ~bits[49];
  wire d3_3_3 = bits[49] & ~bits[50];
  wire d3_4_4 = bits[50] & ~bits[51];
  wire d3_5_5 = bits[51] & ~bits[52];
  wire d3_6_6 = bits[52] & ~bits[53];
  wire d3_7_7 = bits[53] & ~bits[54];
  wire [38:0] hit;
  assign hit[0] = d0_8_8 & d1_10_10 & d2_6_6 & d3_1_1;
  assign hit[1] = d0_4_4 & d1_5_5 & d2_6_6 & d3_2_2;
  assign hit[2] = d0_8_8 & d1_10_10 & d2_6_6 & d3_2_2;
  assign hit[3] = d0_9_9 & d1_10_10 & d2_6_6 & d3_2_2;
  assign hit[4] = d0_9_9 & d1_10_10 & d2_7_7 & d3_2_2;
  assign hit[5] = d0_9_9 & d1_11_11 & d2_7_7 & d3_2_2;
  assign hit[6] = d0_4_4 & d1_5_5 & d2_6_6 & d3_3_3;
  assign hit[7] = d0_9_9 & d1_9_9 & d2_6_6 & d3_3_3;
  assign hit[8] = d0_8_8 & d1_10_10 & d2_6_6 & d3_3_3;
  assign hit[9] = d0_9_9 & d1_10_10 & d2_6_6 & d3_3_3;
  assign hit[10] = d0_9_9 & d1_11_11 & d2_6_6 & d3_3_3;
  assign hit[11] = d0_9_9 & d1_10_10 & d2_7_7 & d3_3_3;
  assign hit[12] = d0_9_9 & d1_11_11 & d2_7_7 & d3_3_3;
  assign hit[13] = d0_9_9 & d1_9_9 & d2_6_6 & d3_4_4;
  assign hit[14] = d0_8_8 & d1_10_10 & d2_6_6 & d3_4_4;
  assign hit[15] = d0_9_9 & d1_10_10 & d2_6_6 & d3_4_4;
  assign hit[16] = d0_9_9 & d1_11_11 & d2_6_6 & d3_4_4;
  assign hit[17] = d0_9_9 & d1_9_9 & d2_7_7 & d3_4_4;
  assign hit[18] = d0_9_9 & d1_10_10 & d2_7_7 & d3_4_4;
  assign hit[19] = d0_9_9 & d1_11_11 & d2_7_7 & d3_4_4;
  assign hit[20] = d0_5_5 & d1_6_6 & d2_6_6 & d3_5_5;
  assign hit[21] = d0_8_8 & d1_10_10 & d2_6_6 & d3_5_5;
  assign hit[22] = d0_9_9 & d1_10_10 & d2_6_6 & d3_5_5;
  assign hit[23] = d0_9_9 & d1_11_11 & d2_6_6 & d3_5_5;
  assign hit[24] = d0_9_9 & d1_9_9 & d2_7_7 & d3_5_5;
  assign hit[25] = d0_9_9 & d1_10_10 & d2_7_7 & d3_5_5;
  assign hit[26] = d0_9_9 & d1_11_11 & d2_7_7 & d3_5_5;
  assign hit[27] = d0_3_3 & d1_5_5 & d2_5_5 & d3_6_6;
  assign hit[28] = d0_3_3 & d1_6_6 & d2_5_5 & d3_6_6;
  assign hit[29] = d0_5_5 & d1_6_6 & d2_6_6 & d3_6_6;
  assign hit[30] = d0_8_8 & d1_10_10 & d2_6_6 & d3_6_6;
  assign hit[31] = d0_9_9 & d1_10_10 & d2_6_6 & d3_6_6;
  assign hit[32] = d0_9_9 & d1_11_11 & d2_6_6 & d3_6_6;
  assign hit[33] = d0_9_9 & d1_10_10 & d2_7_7 & d3_6_6;
  assign hit[34] = d0_9_9 & d1_11_11 & d2_7_7 & d3_6_6;
  assign hit[35] = d0_7_7 & d1_10_10 & d2_6_6 & d3_7_7;
  assign hit[36] = d0_8_8 & d1_10_10 & d2_6_6 & d3_7_7;
  assign hit[37] = d0_9_9 & d1_10_10 & d2_6_6 & d3_7_7;
  assign hit[38] = d0_9_9 & d1_11_11 & d2_6_6 & d3_7_7;
  reg [38:0] hit_r;
  always @(posedge clk or negedge rstn) if (!rstn) hit_r <= 0; else hit_r <= hit;
  wire [5:0] address;
  assign address[0] = hit_r[0] | hit_r[2] | hit_r[4] | hit_r[6] | hit_r[8] | hit_r[10] | hit_r[12] | hit_r[14] | hit_r[16] | hit_r[18] | hit_r[20] | hit_r[22] | hit_r[24] | hit_r[26] | hit_r[28] | hit_r[30] | hit_r[32] | hit_r[34] | hit_r[36] | hit_r[38];
  assign address[1] = hit_r[1] | hit_r[2] | hit_r[5] | hit_r[6] | hit_r[9] | hit_r[10] | hit_r[13] | hit_r[14] | hit_r[17] | hit_r[18] | hit_r[21] | hit_r[22] | hit_r[25] | hit_r[26] | hit_r[29] | hit_r[30] | hit_r[33] | hit_r[34] | hit_r[37] | hit_r[38];
  assign address[2] = hit_r[3] | hit_r[4] | hit_r[5] | hit_r[6] | hit_r[11] | hit_r[12] | hit_r[13] | hit_r[14] | hit_r[19] | hit_r[20] | hit_r[21] | hit_r[22] | hit_r[27] | hit_r[28] | hit_r[29] | hit_r[30] | hit_r[35] | hit_r[36] | hit_r[37] | hit_r[38];
  assign address[3] = hit_r[7] | hit_r[8] | hit_r[9] | hit_r[10] | hit_r[11] | hit_r[12] | hit_r[13] | hit_r[14] | hit_r[23] | hit_r[24] | hit_r[25] | hit_r[26] | hit_r[27] | hit_r[28] | hit_r[29] | hit_r[30];
  assign address[4] = hit_r[15] | hit_r[16] | hit_r[17] | hit_r[18] | hit_r[19] | hit_r[20] | hit_r[21] | hit_r[22] | hit_r[23] | hit_r[24] | hit_r[25] | hit_r[26] | hit_r[27] | hit_r[28] | hit_r[29] | hit_r[30];
  assign address[5] = hit_r[31] | hit_r[32] | hit_r[33] | hit_r[34] | hit_r[35] | hit_r[36] | hit_r[37] | hit_r[38];
  reg [5:0] address_r;
  always @(posedge clk or negedge rstn) if (!rstn) address_r <= 0; else address_r <= address;
  wire [9:0] words;
  (* keep_hierarchy = "yes" *) patch_memory u_memory(.address(address_r), .words(words));
  wire signed [0:0] word0 = words[0 +: 1];
  reg signed [0:0] word0_0;
  always @(posedge clk or negedge rstn) if (!rstn) word0_0 <= 0; else word0_0 <= word0;
  reg signed [0:0] word0_0_1;
  always @(posedge clk or negedge rstn) if (!rstn) word0_0_1 <= 0; else word0_0_1 <= word0_0;
  reg signed [0:0] word0_0_1_2;
  always @(posedge clk or negedge rstn) if (!rstn) word0_0_1_2 <= 0; else word0_0_1_2 <= word0_0_1;
  reg signed [0:0] word0_0_1_2_3;
  always @(posedge clk or negedge rstn) if (!rstn) word0_0_1_2_3 <= 0; else word0_0_1_2_3 <= word0_0_1_2;
  wire signed [9:0] base0 = {{1{1'b0}}, pop[0 +: 9]};
  assign value[0 +: 10] = base0 + word0_0_1_2_3;
  wire signed [8:0] word1 = words[1 +: 9];
  reg signed [8:0] word1_0;
  always @(posedge clk or negedge rstn) if (!rstn) word1_0 <= 0; else word1_0 <= word1;
  reg signed [8:0] word1_0_1;
  always @(posedge clk or negedge rstn) if (!rstn) word1_0_1 <= 0; else word1_0_1 <= word1_0;
  reg signed [8:0] word1_0_1_2;
  always @(posedge clk or negedge rstn) if (!rstn) word1_0_1_2 <= 0; else word1_0_1_2 <= word1_0_1;
  reg signed [8:0] word1_0_1_2_3;
  always @(posedge clk or negedge rstn) if (!rstn) word1_0_1_2_3 <= 0; else word1_0_1_2_3 <= word1_0_1_2;
  wire signed [9:0] base1 = {{1{1'b0}}, pop[9 +: 9]};
  assign value[10 +: 10] = base1 + word1_0_1_2_3;
endmodule

module patch_memory(input [5:0] address, output reg [9:0] words);
  always @* begin
    words = 0;
    case (address)
      1: begin words[1 +: 9] = 32; end
      2: begin words[1 +: 9] = 8; end
      3: begin words[1 +: 9] = 32; end
      4: begin words[1 +: 9] = 8; end
      5: begin words[1 +: 9] = 8; end
      6: begin words[1 +: 9] = 32; end
      7: begin words[1 +: 9] = 32; end
      8: begin words[1 +: 9] = 8; end
      9: begin words[1 +: 9] = 32; end
      10: begin words[1 +: 9] = 32; end
      11: begin words[1 +: 9] = 32; end
      12: begin words[1 +: 9] = 32; end
      13: begin words[1 +: 9] = 159; end
      14: begin words[1 +: 9] = 8; end
      15: begin words[1 +: 9] = 32; end
      16: begin words[1 +: 9] = 32; end
      17: begin words[1 +: 9] = 32; end
      18: begin words[1 +: 9] = 32; end
      19: begin words[1 +: 9] = 32; end
      20: begin words[1 +: 9] = 158; end
      21: begin words[1 +: 9] = 32; end
      22: begin words[1 +: 9] = 8; end
      23: begin words[1 +: 9] = 8; end
      24: begin words[1 +: 9] = 132; end
      25: begin words[1 +: 9] = 32; end
      26: begin words[1 +: 9] = 32; end
      27: begin words[1 +: 9] = 157; end
      28: begin words[1 +: 9] = 8; end
      29: begin words[1 +: 9] = 2; end
      30: begin words[1 +: 9] = 32; end
      31: begin words[1 +: 9] = 8; end
      32: begin words[1 +: 9] = 32; end
      33: begin words[1 +: 9] = 32; end
      34: begin words[1 +: 9] = 137; end
      35: begin words[1 +: 9] = 157; end
      36: begin words[1 +: 9] = 8; end
      37: begin words[1 +: 9] = 8; end
      38: begin words[1 +: 9] = 8; end
      39: begin words[1 +: 9] = 32; end
    endcase
  end
endmodule

module top(input logic clk, input logic rstn, input logic [47:0] x_in, output logic [19:0] out);
  reg [47:0] x;
  always @(posedge clk or negedge rstn) if (!rstn) x <= 0; else x <= x_in;
  logic signed [0:11] quant [0:3];
  assign quant[0] = x[0 +: 12];
  assign quant[1] = x[12 +: 12];
  assign quant[2] = x[24 +: 12];
  assign quant[3] = x[36 +: 12];
  logic [0:75] thermo_bus;
  thermo_encode_thresholded #(.OBS(4), .BITS_PER_OBS(19), .X(12)) u_enc (.quant_in(quant), .thermo_out(thermo_bus));
  wire [55:0] bits;
  assign bits[0] = thermo_bus[0];
  assign bits[1] = thermo_bus[1];
  assign bits[2] = thermo_bus[2];
  assign bits[3] = thermo_bus[3];
  assign bits[4] = thermo_bus[4];
  assign bits[5] = thermo_bus[5];
  assign bits[6] = thermo_bus[6];
  assign bits[7] = thermo_bus[7];
  assign bits[8] = thermo_bus[8];
  assign bits[9] = thermo_bus[9];
  assign bits[10] = thermo_bus[10];
  assign bits[11] = thermo_bus[11];
  assign bits[12] = thermo_bus[12];
  assign bits[13] = thermo_bus[13];
  assign bits[14] = thermo_bus[14];
  assign bits[15] = thermo_bus[15];
  assign bits[16] = thermo_bus[16];
  assign bits[17] = thermo_bus[17];
  assign bits[18] = thermo_bus[18];
  assign bits[19] = thermo_bus[19];
  assign bits[20] = thermo_bus[20];
  assign bits[21] = thermo_bus[21];
  assign bits[22] = thermo_bus[22];
  assign bits[23] = thermo_bus[23];
  assign bits[24] = thermo_bus[24];
  assign bits[25] = thermo_bus[25];
  assign bits[26] = thermo_bus[26];
  assign bits[27] = thermo_bus[27];
  assign bits[28] = thermo_bus[28];
  assign bits[29] = thermo_bus[29];
  assign bits[30] = thermo_bus[30];
  assign bits[31] = thermo_bus[31];
  assign bits[32] = thermo_bus[32];
  assign bits[33] = thermo_bus[33];
  assign bits[34] = thermo_bus[34];
  assign bits[35] = thermo_bus[35];
  assign bits[36] = thermo_bus[36];
  assign bits[37] = thermo_bus[37];
  assign bits[38] = thermo_bus[38];
  assign bits[39] = thermo_bus[39];
  assign bits[40] = thermo_bus[40];
  assign bits[41] = thermo_bus[41];
  assign bits[42] = thermo_bus[42];
  assign bits[43] = thermo_bus[43];
  assign bits[44] = thermo_bus[44];
  assign bits[45] = thermo_bus[45];
  assign bits[46] = thermo_bus[46];
  assign bits[47] = thermo_bus[57];
  assign bits[48] = thermo_bus[58];
  assign bits[49] = thermo_bus[59];
  assign bits[50] = thermo_bus[60];
  assign bits[51] = thermo_bus[61];
  assign bits[52] = thermo_bus[62];
  assign bits[53] = thermo_bus[63];
  assign bits[54] = thermo_bus[64];
  assign bits[55] = thermo_bus[65];
  logic [0:8] group_sum [0:1];
  dwc_core #(.PIPE_STAGES(3)) u_dwc (.clk(clk), .rstn(rstn), .thermo_bus(thermo_bus), .group_sum_int(group_sum));
  wire [17:0] pop;
  assign pop[0 +: 9] = group_sum[0];
  assign pop[9 +: 9] = group_sum[1];
  reg [17:0] pop_r;
  always @(posedge clk or negedge rstn) if (!rstn) pop_r <= 0; else pop_r <= pop;
  wire [19:0] value;
  (* keep_hierarchy = "yes" *) patch u_patch(.clk(clk), .rstn(rstn), .bits(bits), .pop(pop_r), .value(value));
  always @(posedge clk or negedge rstn) if (!rstn) out <= 0; else out <= value;
endmodule
