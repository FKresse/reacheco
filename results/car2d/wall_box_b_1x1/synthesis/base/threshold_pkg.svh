package threshold_pkg;

  localparam int OBS = 4;
  localparam int NUM_THRESHOLDS = 19;
  localparam int THRESH_WIDTH = 12;

  localparam logic signed [THRESH_WIDTH-1:0] NORMALIZATION_THRESHOLDS [0:OBS-1][0:NUM_THRESHOLDS-1] = '{
    '{-12'sd1842, -12'sd1638, -12'sd1433, -12'sd1228, -12'sd1024, -12'sd819, -12'sd614, -12'sd409, -12'sd205, 12'sd0, 12'sd205, 12'sd409, 12'sd614, 12'sd819, 12'sd1024, 12'sd1228, 12'sd1433, 12'sd1638, 12'sd1842},
    '{-12'sd1842, -12'sd1638, -12'sd1433, -12'sd1228, -12'sd1024, -12'sd819, -12'sd614, -12'sd409, -12'sd205, 12'sd0, 12'sd205, 12'sd409, 12'sd614, 12'sd819, 12'sd1024, 12'sd1228, 12'sd1433, 12'sd1638, 12'sd1842},
    '{-12'sd1638, -12'sd1228, -12'sd819, -12'sd409, 12'sd0, 12'sd409, 12'sd819, 12'sd1228, 12'sd1638, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047},
    '{-12'sd1638, -12'sd1228, -12'sd819, -12'sd409, 12'sd0, 12'sd409, 12'sd819, 12'sd1228, 12'sd1638, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047, 12'sd2047}
  };

endpackage : threshold_pkg
