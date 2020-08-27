#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]

computecoeffts[Log[2, #]&, {1 + $MachineEpsilon, 2}, 6, 26, 15, 10, "lg2_1-2.txt"];
