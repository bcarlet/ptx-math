#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]];
Needs["EnhancedMinimax`"]

fmt[output_] :=
	MapAt[MantissaExponent[#, 2]&, output, 2]

Print[fmt[computecoeffts[6, 25, 15, 11]]]
