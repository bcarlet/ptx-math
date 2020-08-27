#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]

computecoeffts[(1/Sqrt[#])&, {1, 2}, 6, 26, 17, 11, "rsqrt_1-2.txt"];
computecoeffts[(1/Sqrt[2 #])&, {1, 2}, 6, 26, 17, 11, "rsqrt2x_1-2.txt"];
