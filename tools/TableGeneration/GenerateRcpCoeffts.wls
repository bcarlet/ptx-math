#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]

computecoeffts[(1/#)&, {1, 2}, 7, 26, 16, 10, "rcp_1-2.txt"];
