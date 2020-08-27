#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]

computecoeffts[Sqrt, {1, 2}, 6, 25, 17, 12, "sqrt_1-2.txt"];
computecoeffts[Sqrt[2 #]&, {1, 2}, 6, 25, 17, 12, "sqrt2x_1-2.txt"];
