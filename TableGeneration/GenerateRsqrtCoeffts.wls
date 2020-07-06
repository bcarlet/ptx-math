#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]
Needs["ErrorOutput`"]

printerror[computecoeffts[(1/Sqrt[#])&, {1, 2}, 6, 26, 17, 11, "rsqrt_1_2.txt"]];
printerror[computecoeffts[(1/Sqrt[2 #])&, {1, 2}, 6, 26, 17, 11, "rsqrt2x_1_2.txt"]];
