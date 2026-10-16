read_verilog -sv {core_pkg.svh threshold_pkg.svh /nfs/scistore16/tomgrp/fkresse/difflogic/test/reacheco/circuit/dwc_hdl/thermo_encode_thresholded.sv /nfs/scistore16/tomgrp/fkresse/difflogic/test/reacheco/circuit/dwc_hdl/mapping_layer.sv /nfs/scistore16/tomgrp/fkresse/difflogic/test/reacheco/circuit/dwc_hdl/dwn_lut_layer.sv /nfs/scistore16/tomgrp/fkresse/difflogic/test/reacheco/circuit/dwc_hdl/dwn_group_sum_pipelined.sv /nfs/scistore16/tomgrp/fkresse/difflogic/test/reacheco/circuit/dwc_hdl/dwc_core.sv design.sv}
synth_design -top top -part xc7a100tfgg484-1 -mode out_of_context -include_dirs [list .]
create_clock -name clk -period 10.0 [get_ports clk]
opt_design
place_design
route_design
report_utilization -file utilization.rpt
report_timing_summary -file timing.rpt
set slack [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set f [open slack.txt w]; puts $f $slack; close $f

