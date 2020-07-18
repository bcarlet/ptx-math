#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]

computecoeffts[(2^#)&, {0, 1}, 6, 25, 15, 11, "ex2_0-1.txt"];
