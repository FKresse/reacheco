`timescale 1ns/1ps
`include "core_pkg.svh"

module patch(input clk, input rstn, input [161:0] bits, input [7:0] pop, output [8:0] value);
  wire d0_1_1 = bits[0] & ~bits[1];
  wire d0_2_2 = bits[1] & ~bits[2];
  wire d0_16_16 = bits[15] & ~bits[16];
  wire d0_17_17 = bits[16] & ~bits[17];
  wire d0_18_18 = bits[17] & ~bits[18];
  wire d0_20_20 = bits[19] & ~bits[20];
  wire d0_21_21 = bits[20] & ~bits[21];
  wire d0_22_22 = bits[21] & ~bits[22];
  wire d0_23_23 = bits[22] & ~bits[23];
  wire d0_24_24 = bits[23] & ~bits[24];
  wire d0_25_25 = bits[24] & ~bits[25];
  wire d0_26_26 = bits[25] & ~bits[26];
  wire d0_27_27 = bits[26] & ~bits[27];
  wire d0_28_28 = bits[27] & ~bits[28];
  wire d0_29_29 = bits[28] & ~bits[29];
  wire d0_30_30 = bits[29] & ~bits[30];
  wire d0_31_31 = bits[30] & ~bits[31];
  wire d0_32_32 = bits[31] & ~bits[32];
  wire d0_33_33 = bits[32] & ~bits[33];
  wire d0_34_34 = bits[33] & ~bits[34];
  wire d0_35_35 = bits[34] & ~bits[35];
  wire d0_36_36 = bits[35] & ~bits[36];
  wire d0_37_37 = bits[36] & ~bits[37];
  wire d0_38_38 = bits[37] & ~bits[38];
  wire d0_39_39 = bits[38] & ~bits[39];
  wire d0_40_40 = bits[39] & ~bits[40];
  wire d1_12_12 = bits[92] & ~bits[93];
  wire d1_13_13 = bits[93] & ~bits[94];
  wire d1_14_14 = bits[94] & ~bits[95];
  wire d1_15_15 = bits[95] & ~bits[96];
  wire d1_16_16 = bits[96] & ~bits[97];
  wire d1_17_17 = bits[97] & ~bits[98];
  wire d1_18_18 = bits[98] & ~bits[99];
  wire d1_19_19 = bits[99] & ~bits[100];
  wire d1_20_20 = bits[100] & ~bits[101];
  wire d1_21_21 = bits[101] & ~bits[102];
  wire d1_22_22 = bits[102] & ~bits[103];
  wire d1_23_23 = bits[103] & ~bits[104];
  wire d1_36_36 = bits[116] & ~bits[117];
  wire d1_37_37 = bits[117] & ~bits[118];
  wire d1_38_38 = bits[118] & ~bits[119];
  wire d1_39_39 = bits[119] & ~bits[120];
  wire d1_40_40 = bits[120] & ~bits[121];
  wire [53:0] hit;
  assign hit[0] = d0_40_40 & d1_12_12;
  assign hit[1] = d0_37_37 & d1_13_13;
  assign hit[2] = d0_38_38 & d1_13_13;
  assign hit[3] = d0_39_39 & d1_13_13;
  assign hit[4] = d0_40_40 & d1_13_13;
  assign hit[5] = d0_34_34 & d1_14_14;
  assign hit[6] = d0_35_35 & d1_14_14;
  assign hit[7] = d0_36_36 & d1_14_14;
  assign hit[8] = d0_37_37 & d1_14_14;
  assign hit[9] = d0_38_38 & d1_14_14;
  assign hit[10] = d0_39_39 & d1_14_14;
  assign hit[11] = d0_40_40 & d1_14_14;
  assign hit[12] = d0_32_32 & d1_15_15;
  assign hit[13] = d0_33_33 & d1_15_15;
  assign hit[14] = d0_34_34 & d1_15_15;
  assign hit[15] = d0_35_35 & d1_15_15;
  assign hit[16] = d0_36_36 & d1_15_15;
  assign hit[17] = d0_37_37 & d1_15_15;
  assign hit[18] = d0_29_29 & d1_16_16;
  assign hit[19] = d0_30_30 & d1_16_16;
  assign hit[20] = d0_31_31 & d1_16_16;
  assign hit[21] = d0_32_32 & d1_16_16;
  assign hit[22] = d0_33_33 & d1_16_16;
  assign hit[23] = d0_34_34 & d1_16_16;
  assign hit[24] = d0_27_27 & d1_17_17;
  assign hit[25] = d0_28_28 & d1_17_17;
  assign hit[26] = d0_29_29 & d1_17_17;
  assign hit[27] = d0_30_30 & d1_17_17;
  assign hit[28] = d0_31_31 & d1_17_17;
  assign hit[29] = d0_32_32 & d1_17_17;
  assign hit[30] = d0_25_25 & d1_18_18;
  assign hit[31] = d0_26_26 & d1_18_18;
  assign hit[32] = d0_27_27 & d1_18_18;
  assign hit[33] = d0_28_28 & d1_18_18;
  assign hit[34] = d0_29_29 & d1_18_18;
  assign hit[35] = d0_23_23 & d1_19_19;
  assign hit[36] = d0_24_24 & d1_19_19;
  assign hit[37] = d0_25_25 & d1_19_19;
  assign hit[38] = d0_26_26 & d1_19_19;
  assign hit[39] = d0_21_21 & d1_20_20;
  assign hit[40] = d0_22_22 & d1_20_20;
  assign hit[41] = d0_23_23 & d1_20_20;
  assign hit[42] = d0_24_24 & d1_20_20;
  assign hit[43] = d0_20_20 & d1_21_21;
  assign hit[44] = d0_17_17 & d1_22_22;
  assign hit[45] = d0_18_18 & d1_22_22;
  assign hit[46] = d0_16_16 & d1_23_23;
  assign hit[47] = d0_17_17 & d1_23_23;
  assign hit[48] = d0_1_1 & d1_36_36;
  assign hit[49] = d0_2_2 & d1_36_36;
  assign hit[50] = d0_1_1 & d1_37_37;
  assign hit[51] = d0_1_1 & d1_38_38;
  assign hit[52] = d0_1_1 & d1_39_39;
  assign hit[53] = d0_1_1 & d1_40_40;
  reg [53:0] hit_r;
  always @(posedge clk or negedge rstn) if (!rstn) hit_r <= 0; else hit_r <= hit;
  wire [5:0] address;
  assign address[0] = hit_r[0] | hit_r[2] | hit_r[4] | hit_r[6] | hit_r[8] | hit_r[10] | hit_r[12] | hit_r[14] | hit_r[16] | hit_r[18] | hit_r[20] | hit_r[22] | hit_r[24] | hit_r[26] | hit_r[28] | hit_r[30] | hit_r[32] | hit_r[34] | hit_r[36] | hit_r[38] | hit_r[40] | hit_r[42] | hit_r[44] | hit_r[46] | hit_r[48] | hit_r[50] | hit_r[52];
  assign address[1] = hit_r[1] | hit_r[2] | hit_r[5] | hit_r[6] | hit_r[9] | hit_r[10] | hit_r[13] | hit_r[14] | hit_r[17] | hit_r[18] | hit_r[21] | hit_r[22] | hit_r[25] | hit_r[26] | hit_r[29] | hit_r[30] | hit_r[33] | hit_r[34] | hit_r[37] | hit_r[38] | hit_r[41] | hit_r[42] | hit_r[45] | hit_r[46] | hit_r[49] | hit_r[50] | hit_r[53];
  assign address[2] = hit_r[3] | hit_r[4] | hit_r[5] | hit_r[6] | hit_r[11] | hit_r[12] | hit_r[13] | hit_r[14] | hit_r[19] | hit_r[20] | hit_r[21] | hit_r[22] | hit_r[27] | hit_r[28] | hit_r[29] | hit_r[30] | hit_r[35] | hit_r[36] | hit_r[37] | hit_r[38] | hit_r[43] | hit_r[44] | hit_r[45] | hit_r[46] | hit_r[51] | hit_r[52] | hit_r[53];
  assign address[3] = hit_r[7] | hit_r[8] | hit_r[9] | hit_r[10] | hit_r[11] | hit_r[12] | hit_r[13] | hit_r[14] | hit_r[23] | hit_r[24] | hit_r[25] | hit_r[26] | hit_r[27] | hit_r[28] | hit_r[29] | hit_r[30] | hit_r[39] | hit_r[40] | hit_r[41] | hit_r[42] | hit_r[43] | hit_r[44] | hit_r[45] | hit_r[46];
  assign address[4] = hit_r[15] | hit_r[16] | hit_r[17] | hit_r[18] | hit_r[19] | hit_r[20] | hit_r[21] | hit_r[22] | hit_r[23] | hit_r[24] | hit_r[25] | hit_r[26] | hit_r[27] | hit_r[28] | hit_r[29] | hit_r[30] | hit_r[47] | hit_r[48] | hit_r[49] | hit_r[50] | hit_r[51] | hit_r[52] | hit_r[53];
  assign address[5] = hit_r[31] | hit_r[32] | hit_r[33] | hit_r[34] | hit_r[35] | hit_r[36] | hit_r[37] | hit_r[38] | hit_r[39] | hit_r[40] | hit_r[41] | hit_r[42] | hit_r[43] | hit_r[44] | hit_r[45] | hit_r[46] | hit_r[47] | hit_r[48] | hit_r[49] | hit_r[50] | hit_r[51] | hit_r[52] | hit_r[53];
  reg [5:0] address_r;
  always @(posedge clk or negedge rstn) if (!rstn) address_r <= 0; else address_r <= address;
  wire [6:0] words;
  (* keep_hierarchy = "yes" *) patch_memory u_memory(.address(address_r), .words(words));
  wire signed [6:0] word0 = words[0 +: 7];
  reg signed [6:0] word0_0;
  always @(posedge clk or negedge rstn) if (!rstn) word0_0 <= 0; else word0_0 <= word0;
  reg signed [6:0] word0_0_1;
  always @(posedge clk or negedge rstn) if (!rstn) word0_0_1 <= 0; else word0_0_1 <= word0_0;
  reg signed [6:0] word0_0_1_2;
  always @(posedge clk or negedge rstn) if (!rstn) word0_0_1_2 <= 0; else word0_0_1_2 <= word0_0_1;
  reg signed [6:0] word0_0_1_2_3;
  always @(posedge clk or negedge rstn) if (!rstn) word0_0_1_2_3 <= 0; else word0_0_1_2_3 <= word0_0_1_2;
  wire signed [8:0] base0 = {{1{1'b0}}, pop[0 +: 8]};
  assign value[0 +: 9] = base0 + word0_0_1_2_3;
endmodule

module patch_memory(input [5:0] address, output reg [6:0] words);
  always @* begin
    words = 0;
    case (address)
      1: begin words[0 +: 7] = 53; end
      2: begin words[0 +: 7] = 45; end
      3: begin words[0 +: 7] = 45; end
      4: begin words[0 +: 7] = 53; end
      5: begin words[0 +: 7] = 55; end
      6: begin words[0 +: 7] = 46; end
      7: begin words[0 +: 7] = 44; end
      8: begin words[0 +: 7] = 44; end
      9: begin words[0 +: 7] = 46; end
      10: begin words[0 +: 7] = 31; end
      11: begin words[0 +: 7] = 22; end
      12: begin words[0 +: 7] = 25; end
      13: begin words[0 +: 7] = 44; end
      14: begin words[0 +: 7] = 41; end
      15: begin words[0 +: 7] = 42; end
      16: begin words[0 +: 7] = 41; end
      17: begin words[0 +: 7] = 26; end
      18: begin words[0 +: 7] = 13; end
      19: begin words[0 +: 7] = 44; end
      20: begin words[0 +: 7] = 44; end
      21: begin words[0 +: 7] = 44; end
      22: begin words[0 +: 7] = 47; end
      23: begin words[0 +: 7] = 28; end
      24: begin words[0 +: 7] = 14; end
      25: begin words[0 +: 7] = 36; end
      26: begin words[0 +: 7] = 36; end
      27: begin words[0 +: 7] = 41; end
      28: begin words[0 +: 7] = 41; end
      29: begin words[0 +: 7] = 26; end
      30: begin words[0 +: 7] = 14; end
      31: begin words[0 +: 7] = 37; end
      32: begin words[0 +: 7] = 38; end
      33: begin words[0 +: 7] = 35; end
      34: begin words[0 +: 7] = 20; end
      35: begin words[0 +: 7] = 10; end
      36: begin words[0 +: 7] = 30; end
      37: begin words[0 +: 7] = 31; end
      38: begin words[0 +: 7] = 32; end
      39: begin words[0 +: 7] = 18; end
      40: begin words[0 +: 7] = 27; end
      41: begin words[0 +: 7] = 27; end
      42: begin words[0 +: 7] = 31; end
      43: begin words[0 +: 7] = 17; end
      44: begin words[0 +: 7] = 13; end
      45: begin words[0 +: 7] = 13; end
      46: begin words[0 +: 7] = 15; end
      47: begin words[0 +: 7] = 14; end
      48: begin words[0 +: 7] = 13; end
      49: begin words[0 +: 7] = 17; end
      50: begin words[0 +: 7] = 17; end
      51: begin words[0 +: 7] = 26; end
      52: begin words[0 +: 7] = 45; end
      53: begin words[0 +: 7] = 35; end
      54: begin words[0 +: 7] = 26; end
    endcase
  end
endmodule

module top(input logic clk, input logic rstn, input logic [23:0] x_in, output logic [8:0] out);
  reg [23:0] x;
  always @(posedge clk or negedge rstn) if (!rstn) x <= 0; else x <= x_in;
  logic signed [0:11] quant [0:1];
  assign quant[0] = x[0 +: 12];
  assign quant[1] = x[12 +: 12];
  logic [0:161] thermo_bus;
  thermo_encode_thresholded #(.OBS(2), .BITS_PER_OBS(81), .X(12)) u_enc (.quant_in(quant), .thermo_out(thermo_bus));
  wire [161:0] bits;
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
  assign bits[47] = thermo_bus[47];
  assign bits[48] = thermo_bus[48];
  assign bits[49] = thermo_bus[49];
  assign bits[50] = thermo_bus[50];
  assign bits[51] = thermo_bus[51];
  assign bits[52] = thermo_bus[52];
  assign bits[53] = thermo_bus[53];
  assign bits[54] = thermo_bus[54];
  assign bits[55] = thermo_bus[55];
  assign bits[56] = thermo_bus[56];
  assign bits[57] = thermo_bus[57];
  assign bits[58] = thermo_bus[58];
  assign bits[59] = thermo_bus[59];
  assign bits[60] = thermo_bus[60];
  assign bits[61] = thermo_bus[61];
  assign bits[62] = thermo_bus[62];
  assign bits[63] = thermo_bus[63];
  assign bits[64] = thermo_bus[64];
  assign bits[65] = thermo_bus[65];
  assign bits[66] = thermo_bus[66];
  assign bits[67] = thermo_bus[67];
  assign bits[68] = thermo_bus[68];
  assign bits[69] = thermo_bus[69];
  assign bits[70] = thermo_bus[70];
  assign bits[71] = thermo_bus[71];
  assign bits[72] = thermo_bus[72];
  assign bits[73] = thermo_bus[73];
  assign bits[74] = thermo_bus[74];
  assign bits[75] = thermo_bus[75];
  assign bits[76] = thermo_bus[76];
  assign bits[77] = thermo_bus[77];
  assign bits[78] = thermo_bus[78];
  assign bits[79] = thermo_bus[79];
  assign bits[80] = thermo_bus[80];
  assign bits[81] = thermo_bus[81];
  assign bits[82] = thermo_bus[82];
  assign bits[83] = thermo_bus[83];
  assign bits[84] = thermo_bus[84];
  assign bits[85] = thermo_bus[85];
  assign bits[86] = thermo_bus[86];
  assign bits[87] = thermo_bus[87];
  assign bits[88] = thermo_bus[88];
  assign bits[89] = thermo_bus[89];
  assign bits[90] = thermo_bus[90];
  assign bits[91] = thermo_bus[91];
  assign bits[92] = thermo_bus[92];
  assign bits[93] = thermo_bus[93];
  assign bits[94] = thermo_bus[94];
  assign bits[95] = thermo_bus[95];
  assign bits[96] = thermo_bus[96];
  assign bits[97] = thermo_bus[97];
  assign bits[98] = thermo_bus[98];
  assign bits[99] = thermo_bus[99];
  assign bits[100] = thermo_bus[100];
  assign bits[101] = thermo_bus[101];
  assign bits[102] = thermo_bus[102];
  assign bits[103] = thermo_bus[103];
  assign bits[104] = thermo_bus[104];
  assign bits[105] = thermo_bus[105];
  assign bits[106] = thermo_bus[106];
  assign bits[107] = thermo_bus[107];
  assign bits[108] = thermo_bus[108];
  assign bits[109] = thermo_bus[109];
  assign bits[110] = thermo_bus[110];
  assign bits[111] = thermo_bus[111];
  assign bits[112] = thermo_bus[112];
  assign bits[113] = thermo_bus[113];
  assign bits[114] = thermo_bus[114];
  assign bits[115] = thermo_bus[115];
  assign bits[116] = thermo_bus[116];
  assign bits[117] = thermo_bus[117];
  assign bits[118] = thermo_bus[118];
  assign bits[119] = thermo_bus[119];
  assign bits[120] = thermo_bus[120];
  assign bits[121] = thermo_bus[121];
  assign bits[122] = thermo_bus[122];
  assign bits[123] = thermo_bus[123];
  assign bits[124] = thermo_bus[124];
  assign bits[125] = thermo_bus[125];
  assign bits[126] = thermo_bus[126];
  assign bits[127] = thermo_bus[127];
  assign bits[128] = thermo_bus[128];
  assign bits[129] = thermo_bus[129];
  assign bits[130] = thermo_bus[130];
  assign bits[131] = thermo_bus[131];
  assign bits[132] = thermo_bus[132];
  assign bits[133] = thermo_bus[133];
  assign bits[134] = thermo_bus[134];
  assign bits[135] = thermo_bus[135];
  assign bits[136] = thermo_bus[136];
  assign bits[137] = thermo_bus[137];
  assign bits[138] = thermo_bus[138];
  assign bits[139] = thermo_bus[139];
  assign bits[140] = thermo_bus[140];
  assign bits[141] = thermo_bus[141];
  assign bits[142] = thermo_bus[142];
  assign bits[143] = thermo_bus[143];
  assign bits[144] = thermo_bus[144];
  assign bits[145] = thermo_bus[145];
  assign bits[146] = thermo_bus[146];
  assign bits[147] = thermo_bus[147];
  assign bits[148] = thermo_bus[148];
  assign bits[149] = thermo_bus[149];
  assign bits[150] = thermo_bus[150];
  assign bits[151] = thermo_bus[151];
  assign bits[152] = thermo_bus[152];
  assign bits[153] = thermo_bus[153];
  assign bits[154] = thermo_bus[154];
  assign bits[155] = thermo_bus[155];
  assign bits[156] = thermo_bus[156];
  assign bits[157] = thermo_bus[157];
  assign bits[158] = thermo_bus[158];
  assign bits[159] = thermo_bus[159];
  assign bits[160] = thermo_bus[160];
  assign bits[161] = thermo_bus[161];
  logic [0:7] group_sum [0:0];
  dwc_core #(.PIPE_STAGES(3)) u_dwc (.clk(clk), .rstn(rstn), .thermo_bus(thermo_bus), .group_sum_int(group_sum));
  wire [7:0] pop;
  assign pop[0 +: 8] = group_sum[0];
  reg [7:0] pop_r;
  always @(posedge clk or negedge rstn) if (!rstn) pop_r <= 0; else pop_r <= pop;
  wire [8:0] value;
  (* keep_hierarchy = "yes" *) patch u_patch(.clk(clk), .rstn(rstn), .bits(bits), .pop(pop_r), .value(value));
  always @(posedge clk or negedge rstn) if (!rstn) out <= 0; else out <= value;
endmodule
