run("Duplicate...", "duplicate");
run("Enhance Contrast...", "saturated=0 normalize process_all");
run("Median 3D...", "x=4 y=4 z=2");
run("Convert to Mask", "method=Otsu background=Dark calculate black create");
selectWindow("MASK_17H-2.tif");
run("Invert LUT");
run("Watershed", "stack");
run("Erode");

// Merging
run("16-bit");
run("Merge Channels...", "c1=MASK_17H-2.tif c4=17H-1.tif create keep ignore");
run("Next Slice [>]");
run("Enhance Contrast", "saturated=0.35");
