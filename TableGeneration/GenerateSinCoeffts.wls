#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]

computecoeffts[Sin[Pi/2 #]&, {0, 1}, 6, 25, 14, 10, "sin_0-1.txt"];
