run("Duplicate...", "duplicate");
run("Gaussian Blur 3D...", "x=2 y=2 z=2");
setAutoThreshold("Otsu dark");
setOption("BlackBackground", false);
run("Convert to Mask", "method=Otsu background=Dark calculate");
run("Distance Transform 3D");
run("Invert", "stack");