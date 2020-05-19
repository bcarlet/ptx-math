#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[DirectoryName[$InputFileName]]
Needs["EnhancedMinimax`"]

errmax = computecoeffts[6, 25, 15, 11, "sqrt_1_2.txt"];
errmaxfmt = MantissaExponent[errmax, 2];

Print[
	"Max error: " <>
	ToString[errmaxfmt[[1]], InputForm] <>
	" * 2^" <>
	ToString[errmaxfmt[[2]], InputForm]
];

Print["Good bits: " <> ToString[Abs[Log[2, errmax]], InputForm]];
